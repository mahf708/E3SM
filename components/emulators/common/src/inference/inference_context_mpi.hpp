/**
 * @file inference_context_mpi.hpp
 * @brief Build an InferenceContext from a live MPI communicator.
 *
 * Only available when the inference layer is built with MPI support
 * (EMULATOR_HAS_MPI).  Kept apart from inference_context.hpp so that the
 * inference engines, and their unit tests, need no MPI at all.
 */

#ifndef E3SM_EMULATOR_INFERENCE_CONTEXT_MPI_HPP
#define E3SM_EMULATOR_INFERENCE_CONTEXT_MPI_HPP

#include "inference_context.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Fill rank/size and the node-local split from a communicator.
 *
 * @param fortran_comm Fortran handle of the **component** communicator, as a
 *                     component gets it from the driver (`MPI_Comm_c2f` of
 *                     its C communicator, or the handle Fortran already
 *                     holds).  Passing MPI_COMM_WORLD in a multi-component
 *                     run is a mistake: collectives would span components.
 * @param device_id    Accelerator ordinal for this rank, or k_no_device to
 *                     leave it unassigned (see assign_device_round_robin()).
 *
 * The node fields come from `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`, so
 * `node_size` is the number of ranks *of this communicator* on the node, not
 * the number of cores.
 *
 * Collective over `fortran_comm`.
 *
 * @throws InferenceError if MPI is not initialized or the handle is invalid
 */
InferenceContext make_context_from_comm(int fortran_comm,
                                        int device_id = k_no_device);

/**
 * @brief Assign this rank an accelerator by its position on the node.
 *
 * Sets `context.device_id = context.node_rank % devices_per_node`, which is
 * the mapping a job launcher usually implies.
 *
 * @throws InferenceError if devices_per_node < 1, or if more ranks share the
 *         node than there are devices — that case needs a decision (share a
 *         device, or aggregate onto one rank per device) rather than a
 *         silently oversubscribed GPU.
 */
void assign_device_round_robin(InferenceContext &context, int devices_per_node);

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_CONTEXT_MPI_HPP
