/**
 * @file tensor.hpp
 * @brief Named, shaped buffers exchanged with an inference backend.
 */

#ifndef E3SM_EMULATOR_INFERENCE_TENSOR_HPP
#define E3SM_EMULATOR_INFERENCE_TENSOR_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief A named, shaped, row-major buffer of doubles.
 *
 * A Fortran array declared `a(nlev,ncol)` is contiguous in `nlev`, so it
 * corresponds to row-major dims `{ncol, nlev}` here, with the leading
 * dimension as the batch (column) dimension.  Move-only, so an accidental
 * deep copy of a multi-megabyte field cannot hide inside a function
 * signature; use clone() for an explicit one.
 */
class Tensor {
public:
  Tensor() = default;

  /// @brief Allocate and zero-fill a tensor of the given shape.
  Tensor(std::string name, std::vector<std::int64_t> dims);

  /// @brief View writable memory owned by the caller.  Nothing is copied.
  static Tensor view(std::string name, double *data,
                     std::vector<std::int64_t> dims);

  /// @brief View read-only memory owned by the caller.  Nothing is copied.
  static Tensor const_view(std::string name, const double *data,
                           std::vector<std::int64_t> dims);

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) = default;
  Tensor &operator=(Tensor &&) = default;

  /// @brief An owning deep copy.
  Tensor clone() const;

  const std::string &name() const { return m_name; }
  const std::vector<std::int64_t> &dims() const { return m_dims; }
  std::size_t rank() const { return m_dims.size(); }
  std::int64_t dim(std::size_t i) const { return m_dims[i]; }

  /// @brief Total element count (product of the dims; 1 for a scalar).
  std::int64_t size() const { return m_size; }
  std::size_t nbytes() const {
    return static_cast<std::size_t>(m_size) * sizeof(double);
  }

  /// @brief True if this tensor owns the memory it describes.
  ///
  /// Tracked explicitly rather than inferred from the storage, because an
  /// owning tensor of size zero has an empty vector and would otherwise
  /// report that it owns nothing -- which is a different claim.
  bool owns_data() const { return m_owns; }

  /// @brief True if the memory may be written through this tensor.
  bool writable() const { return m_writable; }

  /**
   * @brief Writable pointer to the data.
   *
   * Null for an empty tensor, which is a legal thing to hand a model: a rank
   * owning no columns is normal on a large layout, and inference is
   * collective, so it takes part with zero-length fields.
   *
   * @throws InferenceError if this is a read-only view.
   */
  double *data();

  /// @brief Read-only pointer to the data.  Always valid.
  const double *cdata() const { return m_cdata; }

  /// @brief "name[d0,d1,...]", for error messages.
  std::string to_string() const;

private:
  /// @brief Shared by view() and const_view().
  static Tensor make_view(std::string name, std::vector<std::int64_t> dims,
                          const double *cdata, bool writable);

  std::string m_name;
  std::vector<std::int64_t> m_dims;
  std::int64_t m_size = 0;
  std::vector<double> m_storage; ///< the memory, when this tensor owns it
  double *m_data = nullptr;
  const double *m_cdata = nullptr;
  bool m_owns = false; ///< whether m_storage is where the data lives
  /// Whether writing is allowed, which is not the same question as whether
  /// there is anything to write: an empty writable view has a null pointer.
  bool m_writable = false;
};

/**
 * @brief An ordered set of tensors with distinct names.
 *
 * An ordered `std::vector<Tensor>` that enforces one invariant: names are
 * unique, since the python bridge hands the model a dict keyed by name and a
 * duplicate would silently replace the first.  Order is preserved because a
 * positional model signature has to stay deterministic; append and iterate
 * is the whole interface.
 */
class TensorMap {
public:
  /**
   * @brief Take ownership of a tensor, appending it to the order.
   * @throws InferenceError if a tensor of that name is already present.
   */
  void add(Tensor tensor);

  /// @brief Append a view of writable caller memory.
  void wrap(const std::string &name, double *data,
            std::vector<std::int64_t> dims);

  /// @brief Append a view of read-only caller memory.
  void wrap(const std::string &name, const double *data,
            std::vector<std::int64_t> dims);

  std::size_t size() const { return m_tensors.size(); }

  std::vector<Tensor>::iterator begin() { return m_tensors.begin(); }
  std::vector<Tensor>::iterator end() { return m_tensors.end(); }
  std::vector<Tensor>::const_iterator begin() const { return m_tensors.begin(); }
  std::vector<Tensor>::const_iterator end() const { return m_tensors.end(); }

private:
  std::vector<Tensor> m_tensors;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_TENSOR_HPP
