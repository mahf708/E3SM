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

/// @brief Where a tensor's memory lives.
enum class MemorySpace {
  HOST,   ///< Host memory
  DEVICE, ///< GPU memory (CUDA, or HIP through the same interfaces)
};

/// @brief Memory space of a tensor, and the device it is on, if any.
struct TensorMemory {
  MemorySpace space = MemorySpace::HOST;
  int device = 0; ///< Device index, for MemorySpace::DEVICE
};

/**
 * @brief A named, shaped, row-major buffer of doubles.
 *
 * A Tensor either owns its memory or views memory owned by the caller, so
 * model fields can be handed to a backend without a copy. Elements are
 * always double (E3SM's real(r8)); a backend converts to the model's
 * precision if it needs to.
 *
 * Row-major: a Fortran array a(nlev,ncol) corresponds to dims {ncol, nlev}.
 *
 * A view can also describe strided memory (strides in elements, e.g. a padded
 * field), and memory on a device, so that a backend that can use it in place
 * needs no copy. Owning tensors are always contiguous, on the host. Backends
 * say which memory they accept (InferenceBackend::accepts).
 *
 * Move-only, so a large field is never copied by accident.
 */
class Tensor {
public:
  /// @brief Allocate a zero-filled tensor.
  Tensor(std::string name, std::vector<std::int64_t> dims);

  /**
   * @brief View writable memory owned by the caller.
   * @param strides Element strides, one per dim; empty means contiguous
   * @param memory  Where the memory is (host by default)
   */
  static Tensor view(std::string name, double *data,
                     std::vector<std::int64_t> dims,
                     std::vector<std::int64_t> strides = {},
                     TensorMemory memory = {});

  /// @brief View read-only memory owned by the caller.
  static Tensor const_view(std::string name, const double *data,
                           std::vector<std::int64_t> dims,
                           std::vector<std::int64_t> strides = {},
                           TensorMemory memory = {});

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) = default;
  Tensor &operator=(Tensor &&) = default;

  const std::string &name() const { return m_name; }
  const std::vector<std::int64_t> &dims() const { return m_dims; }

  /// @brief Total element count (product of dims).
  std::int64_t size() const { return m_size; }

  /// @brief Element strides, one per dim (row-major contiguous unless set).
  const std::vector<std::int64_t> &strides() const { return m_strides; }

  /// @brief Whether the elements are contiguous and row-major.
  bool contiguous() const;

  /// @brief Elements between the first and the last, plus one (0 if empty).
  std::int64_t span() const;

  const TensorMemory &memory() const { return m_memory; }
  bool on_device() const { return m_memory.space == MemorySpace::DEVICE; }

  bool writable() const { return m_writable; }

  /**
   * @brief Writable pointer to the data (null if the tensor is empty).
   * @throws InferenceError if this is a read-only view
   */
  double *data();

  /// @brief Read-only pointer to the data (null if the tensor is empty).
  const double *cdata() const { return m_data; }

  /// @brief "name[d0,d1,...]", for messages.
  std::string to_string() const;

private:
  Tensor() = default;

  static Tensor make_view(std::string name, const double *data,
                          std::vector<std::int64_t> dims,
                          std::vector<std::int64_t> strides,
                          TensorMemory memory, bool writable);

  std::string m_name;
  std::vector<std::int64_t> m_dims;
  std::vector<std::int64_t> m_strides;
  TensorMemory m_memory;
  std::int64_t m_size = 0;
  std::vector<double> m_storage; ///< Used only when the tensor owns its data
  const double *m_data = nullptr;
  bool m_writable = false;
};

/**
 * @brief An ordered set of tensors with unique names.
 *
 * Order is preserved because backends may pass tensors to a model
 * positionally.
 */
class TensorMap {
public:
  /**
   * @brief Append a tensor.
   * @throws InferenceError if the name is already present
   */
  void add(Tensor tensor);

  /// @brief Append a view of writable caller memory.
  void wrap(const std::string &name, double *data,
            std::vector<std::int64_t> dims,
            std::vector<std::int64_t> strides = {}, TensorMemory memory = {});

  /// @brief Append a view of read-only caller memory.
  void wrap(const std::string &name, const double *data,
            std::vector<std::int64_t> dims,
            std::vector<std::int64_t> strides = {}, TensorMemory memory = {});

  std::size_t size() const { return m_tensors.size(); }

  std::vector<Tensor>::iterator begin() { return m_tensors.begin(); }
  std::vector<Tensor>::iterator end() { return m_tensors.end(); }
  std::vector<Tensor>::const_iterator begin() const {
    return m_tensors.begin();
  }
  std::vector<Tensor>::const_iterator end() const { return m_tensors.end(); }

private:
  std::vector<Tensor> m_tensors;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_TENSOR_HPP
