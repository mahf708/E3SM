/**
 * @file tensor.hpp
 * @brief Memory-managed data container for inference input/output.
 *
 * Provides Tensor (a typed, shaped buffer that either owns its memory or
 * views memory owned by somebody else) and TensorMap (an ordered,
 * name-addressable collection of Tensors).  These are the only data types
 * that cross the InferenceBackend interface, so a component can hand its
 * own coupling buffers to any backend without an intermediate copy.
 *
 * Conventions:
 *  - Shapes are row-major (C order).  Fortran callers that pass a
 *    column-major array `a(nx,ny)` should declare dims `{ny, nx}`.
 *  - The leading dimension is by convention the batch (column) dimension
 *    and is the one allowed to vary between steps.
 *  - Tensors are move-only: copies are always explicit (clone()), so an
 *    accidental deep copy of a multi-megabyte field cannot hide in a
 *    function signature.
 */

#ifndef E3SM_EMULATOR_INFERENCE_TENSOR_HPP
#define E3SM_EMULATOR_INFERENCE_TENSOR_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <type_traits>
#include <vector>

#include "inference_error.hpp"

namespace emulator {
namespace inference {

// ===========================================================================
// Data types
// ===========================================================================

/**
 * @brief Element types supported by Tensor.
 *
 * E3SM fields are `real(r8)` (FLOAT64) while most trained models expect
 * FLOAT32, so conversion between the two is a first-class operation.
 */
enum class DType {
  FLOAT32, ///< IEEE single precision (C float)
  FLOAT64, ///< IEEE double precision (C double, E3SM's r8)
  INT32,   ///< 32-bit signed integer
  INT64    ///< 64-bit signed integer
};

/// @brief Size in bytes of one element of the given type.
std::size_t dtype_size(DType dtype);

/// @brief Canonical name of a type ("float32", "float64", "int32", "int64").
const char *dtype_name(DType dtype);

/**
 * @brief Parse a type name.
 *
 * Accepts the canonical names plus common aliases: "float"/"f32"/"single"/
 * "real4", "double"/"f64"/"real8", "int"/"i32", "long"/"i64".
 * @throws InferenceError if the name is not recognized.
 */
DType dtype_from_string(const std::string &name);

/// @brief Compile-time map from C++ type to DType (used by typed accessors).
template <typename T> struct dtype_of;
template <> struct dtype_of<float> {
  static constexpr DType value = DType::FLOAT32;
};
template <> struct dtype_of<double> {
  static constexpr DType value = DType::FLOAT64;
};
template <> struct dtype_of<std::int32_t> {
  static constexpr DType value = DType::INT32;
};
template <> struct dtype_of<std::int64_t> {
  static constexpr DType value = DType::INT64;
};

// ===========================================================================
// Tensor
// ===========================================================================

/**
 * @brief A typed, shaped data buffer that may own or view its memory.
 *
 * Two construction modes:
 *  - *owning*: `Tensor t("x", {ncol, nlev}, DType::FLOAT32)` allocates
 *    64-byte-aligned, zero-initialized storage and frees it on destruction.
 *  - *view*:   `Tensor::wrap("x", ptr, {ncol, nlev}, DType::FLOAT64)` refers
 *    to memory owned elsewhere (an MCT coupling buffer, a component's field
 *    array, ...).  Nothing is allocated, copied or freed.
 *
 * A view built from a `const void*` is read-only: data() on a non-const
 * Tensor and any mutating call throw InferenceError.
 */
class Tensor {
public:
  /// @brief Construct an empty tensor (no data, rank 0).
  Tensor() = default;

  /**
   * @brief Construct an owning tensor with zero-initialized storage.
   * @param name  Field name used for name-based lookup and diagnostics
   * @param dims  Row-major extents; all entries must be >= 0
   * @param dtype Element type
   */
  Tensor(std::string name, std::vector<std::int64_t> dims,
         DType dtype = DType::FLOAT64);

  /**
   * @brief Create a writable view of externally owned memory.
   * @param name  Field name
   * @param data  Pointer to at least product(dims) elements of type dtype
   * @param dims  Row-major extents
   * @param dtype Element type of the pointed-to memory
   */
  static Tensor wrap(std::string name, void *data,
                     std::vector<std::int64_t> dims, DType dtype);

  /// @brief Create a read-only view of externally owned memory.
  static Tensor wrap(std::string name, const void *data,
                     std::vector<std::int64_t> dims, DType dtype);

  /// @brief Convenience: writable view whose dtype follows the pointer type.
  template <typename T>
  static Tensor wrap(std::string name, T *data,
                     std::vector<std::int64_t> dims) {
    return wrap(std::move(name), static_cast<void *>(data), std::move(dims),
                dtype_of<T>::value);
  }

  /// @brief Convenience: read-only view whose dtype follows the pointer type.
  template <typename T>
  static Tensor wrap(std::string name, const T *data,
                     std::vector<std::int64_t> dims) {
    return wrap(std::move(name), static_cast<const void *>(data),
                std::move(dims), dtype_of<T>::value);
  }

  ~Tensor();

  // Move-only: see file-level note on explicit copies.
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  /// @brief Deep copy into a new owning tensor (works for views too).
  Tensor clone() const;

  // -------------------------------------------------------------------------
  // Metadata
  // -------------------------------------------------------------------------

  const std::string &name() const { return m_name; }
  void set_name(std::string name) { m_name = std::move(name); }

  DType dtype() const { return m_dtype; }
  const std::vector<std::int64_t> &dims() const { return m_dims; }
  int rank() const { return static_cast<int>(m_dims.size()); }

  /// @brief Extent of dimension i (throws if out of range).
  std::int64_t dim(int i) const;

  /// @brief Total number of elements (1 for a rank-0 tensor with storage).
  std::int64_t size() const { return m_size; }

  /// @brief Total number of bytes of data.
  std::size_t nbytes() const {
    return static_cast<std::size_t>(m_size) * dtype_size(m_dtype);
  }

  bool owns_data() const { return m_owned; }
  bool is_view() const { return m_data != nullptr && !m_owned; }
  bool writable() const { return m_writable; }
  bool empty() const { return m_data == nullptr || m_size == 0; }

  // -------------------------------------------------------------------------
  // Data access
  // -------------------------------------------------------------------------

  /// @brief Raw writable pointer (throws if the tensor is read-only/empty).
  void *data();

  /// @brief Raw read-only pointer (throws if the tensor has no storage).
  const void *data() const;

  /// @brief Typed writable pointer (throws unless dtype_of<T> == dtype()).
  template <typename T> T *data() {
    check_dtype(dtype_of<T>::value);
    return static_cast<T *>(data());
  }

  /// @brief Typed read-only pointer (throws unless dtype_of<T> == dtype()).
  template <typename T> const T *data() const {
    check_dtype(dtype_of<T>::value);
    return static_cast<const T *>(data());
  }

  /**
   * @brief Raw read-only pointer, also on a non-const tensor.
   *
   * `data()` on a non-const tensor asks for write access and therefore
   * throws for a view over const memory.  Use cdata() to read from a tensor
   * you happen to hold by non-const reference.
   */
  const void *cdata() const { return data(); }

  /// @brief Typed read-only pointer, also on a non-const tensor.
  template <typename T> const T *cdata() const {
    check_dtype(dtype_of<T>::value);
    return static_cast<const T *>(data());
  }

  /// @brief Bounds- and type-checked flat element access.
  template <typename T> T &flat(std::int64_t i) {
    check_index(i);
    return data<T>()[i];
  }

  /// @brief Bounds- and type-checked flat element access (const).
  template <typename T> const T &flat(std::int64_t i) const {
    check_index(i);
    return data<T>()[i];
  }

  /// @brief Bounds- and type-checked read-only element access.
  template <typename T> const T &cflat(std::int64_t i) const {
    check_index(i);
    return cdata<T>()[i];
  }

  // -------------------------------------------------------------------------
  // Shape / content manipulation
  // -------------------------------------------------------------------------

  /**
   * @brief Reinterpret the shape without touching memory.
   * @throws InferenceError if the element count would change.
   */
  void reshape(const std::vector<std::int64_t> &dims);

  /**
   * @brief Resize an owning tensor, reallocating only if it must grow.
   *
   * Contents are not preserved.  Intended for per-step batch changes
   * (e.g. a varying number of local columns), where keeping the high-water
   * mark allocation avoids per-step malloc traffic.
   *
   * @throws InferenceError if the tensor is a view.
   */
  void resize(const std::vector<std::int64_t> &dims);

  /// @brief Resize an owning tensor, also changing its element type.
  void resize(const std::vector<std::int64_t> &dims, DType dtype);

  /**
   * @brief Set the leading (batch) extent, resizing an owning tensor.
   *
   * For a view, only the recorded shape changes and the caller is
   * responsible for the memory being large enough; shrinking is always safe.
   */
  void set_batch_size(std::int64_t batch);

  /// @brief Fill all bytes with zero.
  void zero();

  /**
   * @brief Copy element data from another tensor, converting type if needed.
   * @throws InferenceError on element-count mismatch or read-only target.
   */
  void copy_from(const Tensor &src);

  /// @brief Copy `count` host elements in, converting type if needed.
  template <typename T> void copy_from_host(const T *src, std::int64_t count) {
    copy_from(Tensor::wrap("<host>", src, {count}));
  }

  /// @brief Copy `count` elements out to host memory, converting if needed.
  template <typename T> void copy_to_host(T *dst, std::int64_t count) const {
    Tensor tmp = Tensor::wrap("<host>", dst, {count});
    tmp.copy_from(*this);
  }

  /// @brief One-line summary for logs, e.g. `T[10,4]:float32 (owned)`.
  std::string to_string() const;

private:
  void allocate(std::size_t nbytes);
  void release();
  void check_dtype(DType expected) const;
  void check_index(std::int64_t i) const;
  static std::int64_t compute_size(const std::vector<std::int64_t> &dims);

  std::string m_name;
  std::vector<std::int64_t> m_dims;
  DType m_dtype = DType::FLOAT64;
  void *m_data = nullptr;
  std::int64_t m_size = 0;       ///< Element count described by m_dims
  std::size_t m_capacity = 0;    ///< Allocated bytes (owning tensors only)
  bool m_owned = false;          ///< True if this tensor frees m_data
  bool m_writable = false;       ///< False for views over const memory
};

// ===========================================================================
// TensorSpec
// ===========================================================================

/**
 * @brief Declared shape/type of one model input or output.
 *
 * A dim of -1 is dynamic and is filled in with the batch size at run time.
 * Specs come either from configuration (for backends that cannot introspect
 * a model, such as the Python bridge) or from the model itself (ONNX).
 */
struct TensorSpec {
  std::string name;                ///< Tensor name
  std::vector<std::int64_t> dims;  ///< Extents; -1 marks a dynamic dimension
  DType dtype = DType::FLOAT64;    ///< Element type

  TensorSpec() = default;
  TensorSpec(std::string n, std::vector<std::int64_t> d,
             DType t = DType::FLOAT64)
      : name(std::move(n)), dims(std::move(d)), dtype(t) {}

  /**
   * @brief Parse the compact textual form `name[d0,d1,...]:dtype`.
   *
   * The shape and type parts are optional: `q`, `q[-1,72]` and
   * `q[-1,72]:float32` are all valid.  Use -1 for dynamic extents.
   */
  static TensorSpec parse(const std::string &text);

  /// @brief Inverse of parse().
  std::string to_string() const;

  /// @brief Dims with every dynamic (-1) extent replaced by `batch`.
  std::vector<std::int64_t> dims_with_batch(std::int64_t batch) const;

  /// @brief Element count of dims_with_batch(batch).
  std::int64_t size_with_batch(std::int64_t batch) const;

  /// @brief True if the spec has no dynamic extents.
  bool is_static() const;

  /// @brief Allocate an owning tensor matching this spec.
  Tensor make_tensor(std::int64_t batch = 1) const;
};

/**
 * @brief Check a tensor against a spec.
 *
 * Dynamic (-1) spec extents match anything.  A spec with an empty dims list
 * constrains only the name and type.
 *
 * @param spec   Declared spec
 * @param tensor Tensor to check
 * @param why    If non-null, receives a human-readable mismatch reason
 * @return true if the tensor satisfies the spec
 */
bool spec_matches(const TensorSpec &spec, const Tensor &tensor,
                  std::string *why = nullptr);

// ===========================================================================
// TensorMap
// ===========================================================================

/**
 * @brief Ordered, name-addressable set of tensors.
 *
 * Insertion order is preserved because some backends are positional
 * (TorchScript `forward(a, b)`) while others are name-based (ONNX, Python).
 * Names must be unique.  Like Tensor, the map is move-only.
 *
 * References returned by add()/emplace()/wrap()/at() stay valid as more
 * tensors are added — a caller can hold on to one while building up the rest
 * of the map.  erase() and clear() do invalidate them.
 */
class TensorMap {
public:
  TensorMap() = default;
  TensorMap(TensorMap &&) = default;
  TensorMap &operator=(TensorMap &&) = default;
  TensorMap(const TensorMap &) = delete;
  TensorMap &operator=(const TensorMap &) = delete;

  /**
   * @brief Insert a tensor, taking ownership of the handle.
   * @return Reference to the stored tensor
   * @throws InferenceError if the name is empty or already present
   */
  Tensor &add(Tensor tensor);

  /// @brief Create and insert an owning tensor.
  Tensor &emplace(const std::string &name, std::vector<std::int64_t> dims,
                  DType dtype = DType::FLOAT64);

  /// @brief Insert a writable view of external memory.
  Tensor &wrap(const std::string &name, void *data,
               std::vector<std::int64_t> dims, DType dtype);

  /// @brief Insert a read-only view of external memory.
  Tensor &wrap(const std::string &name, const void *data,
               std::vector<std::int64_t> dims, DType dtype);

  /// @brief Insert a view whose dtype follows the pointer type.
  template <typename T>
  Tensor &wrap(const std::string &name, T *data,
               std::vector<std::int64_t> dims) {
    return add(Tensor::wrap(name, data, std::move(dims)));
  }

  /// @brief Insert a read-only view whose dtype follows the pointer type.
  template <typename T>
  Tensor &wrap(const std::string &name, const T *data,
               std::vector<std::int64_t> dims) {
    return add(Tensor::wrap(name, data, std::move(dims)));
  }

  bool has(const std::string &name) const;

  /// @brief Look up by name; throws listing available names if absent.
  Tensor &at(const std::string &name);
  const Tensor &at(const std::string &name) const;

  /// @brief Pointer to the named tensor, or nullptr if absent.
  Tensor *find(const std::string &name);
  const Tensor *find(const std::string &name) const;

  /// @brief Positional access in insertion order.
  Tensor &operator[](std::size_t i);
  const Tensor &operator[](std::size_t i) const;

  std::size_t size() const { return m_tensors.size(); }
  bool empty() const { return m_tensors.empty(); }
  void clear() { m_tensors.clear(); }

  /// @brief Remove one tensor by name; returns true if it was present.
  bool erase(const std::string &name);

  /// @brief Names in insertion order.
  std::vector<std::string> names() const;

  /// @brief Comma-separated names, for diagnostics.
  std::string names_string() const;

  using iterator = std::deque<Tensor>::iterator;
  using const_iterator = std::deque<Tensor>::const_iterator;
  iterator begin() { return m_tensors.begin(); }
  iterator end() { return m_tensors.end(); }
  const_iterator begin() const { return m_tensors.begin(); }
  const_iterator end() const { return m_tensors.end(); }

private:
  // A deque, not a vector: growing the map must not invalidate references
  // that a caller is still holding (see the note above).
  std::deque<Tensor> m_tensors;
};

/// @brief Allocate one owning tensor per spec, sized for `batch`.
TensorMap make_tensors(const std::vector<TensorSpec> &specs,
                       std::int64_t batch = 1);

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_TENSOR_HPP
