/**
 * @file inference_context.hpp
 * @brief The resources a component hands the inference layer.
 *
 * Everything a model needs to know about *where* it is running and *which
 * part of the world it owns*, and nothing about what the model is.  Both
 * halves come from the MCT coupling layer: the component communicator, and
 * the horizontal decomposition the coupler already assigned to this rank.
 *
 * The first half exists because of a specific failure.  A distributed model
 * that discovers its rank from `SLURM_PROCID` / `SLURM_NTASKS` -- which is
 * what ACE's `TorchDistributed` and PhysicsNeMo's `DistributedManager` do by
 * default -- sees the *entire* coupled job, so its process group waits
 * forever for ocean and land ranks that will never join it.  The rendezvous
 * below is built from the component communicator instead, and the python
 * bridge publishes it so an unmodified upstream model initializes over
 * exactly our ranks.
 */

#ifndef E3SM_EMULATOR_INFERENCE_CONTEXT_HPP
#define E3SM_EMULATOR_INFERENCE_CONTEXT_HPP

#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief Ranks, node placement and horizontal decomposition.
 *
 * Default-constructed, it describes a serial run with no grid, which is what
 * tools and unit tests want.
 */
struct InferenceContext {
  // --- parallel resources, from the component communicator ---------------
  int rank = 0;       ///< Rank within the component communicator
  int size = 1;       ///< Size of the component communicator
  int local_rank = 0; ///< Rank among this component's ranks on this node
  int local_size = 1; ///< Number of this component's ranks on this node
  std::string node_name;                 ///< Hostname, for logging and binding
  std::string master_addr = "127.0.0.1"; ///< Rendezvous host (rank 0's)
  int master_port = 0;                   ///< Rendezvous port, 0 if unset

  // --- horizontal decomposition, from the coupler ------------------------
  int nx = 0;                ///< Global longitude points (0 if unstructured)
  int ny = 0;                ///< Global latitude points
  int num_global_cols = 0;   ///< Global column count
  std::vector<int> col_gids; ///< 1-based global ids of this rank's columns
  std::vector<double> lat;   ///< Latitude of each local column [degrees]
  std::vector<double> lon;   ///< Longitude of each local column [degrees]

  int num_local_cols() const { return static_cast<int>(col_gids.size()); }

  /// @brief True if this rank is the one that speaks for the component.
  bool is_root() const { return rank == 0; }

  /// @brief Store the decomposition the coupler gave this rank.
  void set_grid(int nx_, int ny_, int num_global_cols_, const int *gids,
                const double *lat_, const double *lon_, int num_local_cols_);

  /// @brief One-line summary, for logs.
  std::string to_string() const;
};

/**
 * @brief Build a context from a Fortran MPI communicator handle.
 *
 * Fills rank/size from the communicator, local_rank/local_size by splitting
 * it across shared-memory domains (which is what decides GPU affinity), and
 * establishes the rendezvous: rank 0 picks its hostname and a free TCP port
 * and broadcasts both.
 *
 * Built without MPI -- or under MPI but never launched, as a unit test is --
 * this returns a serial context, so the same call site works either way.
 *
 * @param fcomm Fortran communicator handle, as passed through the MCT layer.
 */
InferenceContext make_context(int fcomm);

/**
 * @brief Throw on every rank of `fcomm`, or on none.
 *
 * A configuration mistake usually holds on one rank only -- a decomposition
 * that leaves one rank a different column count, a field the coupler drops
 * on one side.  If that rank throws alone, the others sail on into
 * collective model initialization and the run *hangs* rather than failing,
 * which is far harder to diagnose than the original mistake.
 *
 * So every rank reports whether it is unhappy, the answer is reduced over
 * the communicator, and either all of them throw or none does.  A rank with
 * a complaint raises it; the others say which neighbour to look at.
 *
 * **This is itself a collective**, so it must be reached exactly once on
 * every rank: callers collect their complaints rather than throwing them.
 *
 * @param fcomm   Fortran communicator handle to agree over.
 * @param problem This rank's complaint, or "" if it has none.
 * @throws InferenceError if any rank has a complaint.
 */
void agree_or_throw(int fcomm, const std::string &problem);

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_CONTEXT_HPP
