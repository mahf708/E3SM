/**
 * @file inference_context.hpp
 * @brief Where a backend runs: which rank, which node, which accelerator.
 *
 * This header deliberately does not include <mpi.h>.  The communicator is
 * carried as a Fortran integer handle so that the inference engines and their
 * unit tests stay free of an MPI dependency; the MPI-aware helpers live in
 * inference_context_mpi.hpp and are compiled separately.
 */

#ifndef E3SM_EMULATOR_INFERENCE_CONTEXT_HPP
#define E3SM_EMULATOR_INFERENCE_CONTEXT_HPP

#include <string>

namespace emulator {
namespace inference {

/// @brief Value meaning "no communicator" in InferenceContext::comm.
constexpr int k_no_comm = -1;

/// @brief Value meaning "no accelerator assigned" in InferenceContext.
constexpr int k_no_device = -1;

/**
 * @brief The parallel environment an executor runs in.
 *
 * A component already knows all of this — EmulatorAtm has its component
 * communicator, its rank and its decomposition — and passing it down is what
 * lets the inference layer make correct decisions about device assignment,
 * thread counts and which ranks may write diagnostics.
 *
 * `comm` is the **component** communicator, not MPI_COMM_WORLD: in an E3SM
 * run the atmosphere's ranks are a subset of the world, and a collective on
 * the wrong communicator hangs.
 *
 * Build one with `InferenceContext::serial()` for a single-process run, or
 * with `make_context_from_comm()` (inference_context_mpi.hpp) to fill the
 * rank/node fields from a live communicator.
 */
struct InferenceContext {
  /// Fortran handle of the component communicator, or k_no_comm.
  int comm = k_no_comm;

  int rank = 0; ///< Rank within `comm`
  int size = 1; ///< Number of ranks in `comm`

  /// Rank within this node's subset of `comm` (shared-memory split).
  int node_rank = 0;
  /// Number of ranks of `comm` on this node.
  int node_size = 1;

  /// Accelerator ordinal this rank should use, or k_no_device.
  int device_id = k_no_device;

  /**
   * @brief Execution stream this rank's work should join, or nullptr.
   *
   * Untyped on purpose (`cudaStream_t`/`hipStream_t`/`sycl::queue*`); a
   * backend that understands the build's programming model casts it.  Unused
   * today — reserved so that adding device-resident tensors does not have to
   * change this struct.
   */
  void *stream = nullptr;

  /// @brief Context for a single-process run.
  static InferenceContext serial() { return InferenceContext(); }

  /// @brief True on rank 0 of the component communicator.
  bool is_root() const { return rank == 0; }

  /// @brief True on the first rank of `comm` present on this node.
  bool is_node_root() const { return node_rank == 0; }

  /// @brief True if this context describes more than one rank.
  bool is_parallel() const { return size > 1; }

  /// @brief One-line summary for logs.
  std::string to_string() const;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_CONTEXT_HPP
