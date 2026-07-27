/**
 * @file inference_context_mpi.cpp
 * @brief InferenceContext construction from an MPI communicator.
 *
 * The only translation unit in the inference layer that includes <mpi.h>.
 */

#include "inference_context_mpi.hpp"

#include <mpi.h>

#include "inference_error.hpp"

namespace emulator {
namespace inference {

InferenceContext make_context_from_comm(int fortran_comm, int device_id) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  EMULATOR_INFER_REQUIRE(initialized != 0,
                         "make_context_from_comm() was called before MPI was "
                         "initialized. Build the inference context after the "
                         "component has its communicator.");

  const MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_comm));
  EMULATOR_INFER_REQUIRE(comm != MPI_COMM_NULL,
                         "make_context_from_comm() got a null communicator "
                         "(fortran handle "
                             << fortran_comm << ").");

  InferenceContext context;
  context.comm = fortran_comm;
  context.device_id = device_id;

  EMULATOR_INFER_REQUIRE(MPI_Comm_rank(comm, &context.rank) == MPI_SUCCESS,
                         "MPI_Comm_rank failed on the communicator passed to "
                         "make_context_from_comm().");
  EMULATOR_INFER_REQUIRE(MPI_Comm_size(comm, &context.size) == MPI_SUCCESS,
                         "MPI_Comm_size failed on the communicator passed to "
                         "make_context_from_comm().");

  // Node-local split: how many ranks of *this* communicator share a node, and
  // where this rank sits among them.  That is what device assignment and
  // one-instance-per-node policies need.
  MPI_Comm node_comm = MPI_COMM_NULL;
  const int split_status = MPI_Comm_split_type(
      comm, MPI_COMM_TYPE_SHARED, context.rank, MPI_INFO_NULL, &node_comm);
  if (split_status == MPI_SUCCESS && node_comm != MPI_COMM_NULL) {
    MPI_Comm_rank(node_comm, &context.node_rank);
    MPI_Comm_size(node_comm, &context.node_size);
    MPI_Comm_free(&node_comm);
  } else {
    // Without a shared-memory split we cannot tell ranks on a node apart;
    // report the single-rank-per-node case rather than guessing.
    context.node_rank = 0;
    context.node_size = 1;
  }

  return context;
}

void assign_device_round_robin(InferenceContext &context,
                               int devices_per_node) {
  EMULATOR_INFER_REQUIRE(devices_per_node >= 1,
                         "assign_device_round_robin() needs at least one "
                         "device per node (got "
                             << devices_per_node << ").");
  EMULATOR_INFER_REQUIRE(
      context.node_size <= devices_per_node,
      "There are " << context.node_size << " ranks of this component on the "
                   << "node but only " << devices_per_node
                   << " device(s). Sharing a device between ranks needs an "
                      "explicit decision: give the component one rank per "
                      "device, or aggregate onto one rank per device (an "
                      "execution policy that is not implemented yet). Set "
                      "device_id by hand to override.");
  context.device_id = context.node_rank % devices_per_node;
}

} // namespace inference
} // namespace emulator
