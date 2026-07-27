/**
 * @file test_inference_context_mpi.cpp
 * @brief The one test in this layer that actually runs on several ranks.
 *
 * It checks that a component communicator survives the trip into the
 * inference layer, and that a local-replica executor does the right thing
 * when each rank owns a different number of columns.  Run under mpirun; the
 * CMake target sets that up.
 *
 * Catch2's main is not used here: the ranks must agree on MPI_Init/Finalize,
 * and a plain main keeps the failure reporting per-rank and obvious.
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <mpi.h>

#include "inference_context_mpi.hpp"
#include "inference_executor.hpp"

using namespace emulator::inference;

namespace {

int g_rank = 0;
int g_failures = 0;

void check(bool condition, const std::string &what) {
  if (!condition) {
    std::fprintf(stderr, "[rank %d] FAILED: %s\n", g_rank, what.c_str());
    ++g_failures;
  }
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // A component never gets MPI_COMM_WORLD: it gets its own communicator, and
  // collectives on the wrong one hang.  Split so the test exercises that.
  MPI_Comm component_comm = MPI_COMM_NULL;
  MPI_Comm_dup(MPI_COMM_WORLD, &component_comm);
  const int fortran_comm = static_cast<int>(MPI_Comm_c2f(component_comm));

  try {
    const InferenceContext context = make_context_from_comm(fortran_comm);

    check(context.rank == g_rank, "context rank matches the communicator");
    check(context.size == world_size, "context size matches the communicator");
    check(context.comm == fortran_comm, "context keeps the component handle");
    check(context.node_size >= 1 && context.node_size <= world_size,
          "node size is within range");
    check(context.node_rank >= 0 && context.node_rank < context.node_size,
          "node rank is within range");
    check(context.is_root() == (g_rank == 0), "exactly rank 0 is root");
    check(context.is_parallel() == (world_size > 1),
          "is_parallel reflects the communicator");

    // Every rank builds its own replica and infers on its own columns; the
    // counts differ on purpose, which is the normal E3SM situation.
    InferenceConfig config;
    config.set("mode", std::string("affine"))
        .set("scale", 2.0)
        .set("offset", static_cast<double>(g_rank));
    config.outputs.push_back(TensorSpec("y", {-1, 2}, DType::FLOAT64));
    config.verbose = true; // only rank 0 should actually print

    auto executor = create_and_init_executor(config, context);
    check(executor->policy() == ExecutionPolicy::LOCAL_REPLICA,
          "default policy is local_replica");
    check(executor->owns_model(), "every rank owns a replica");
    check(executor->backend().config().verbose == (g_rank == 0),
          "only rank 0 is verbose");

    const std::int64_t local_columns = 2 + g_rank; // deliberately ragged
    TensorMap inputs;
    Tensor &x = inputs.emplace("x", {local_columns, 2});
    for (std::int64_t i = 0; i < x.size(); ++i) {
      x.flat<double>(i) = static_cast<double>(i);
    }
    TensorMap outputs;

    check(executor->infer(inputs, outputs), "inference succeeds on this rank");
    check(outputs.at("y").dim(0) == local_columns,
          "output carries this rank's column count");
    check(outputs.at("y").flat<double>(1) == 2.0 + static_cast<double>(g_rank),
          "each rank ran its own replica");

    executor->finalize();

    // The device rules should be enforced from a real node-local split too.
    if (context.node_size > 1) {
      InferenceConfig gpu_config;
      gpu_config.set("device", std::string("cuda"));
      bool threw = false;
      try {
        create_executor(gpu_config, context);
      } catch (const InferenceError &) {
        threw = true;
      }
      check(threw, "sharing a node without a device assignment is refused");
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[rank %d] unexpected exception: %s\n", g_rank,
                 e.what());
    ++g_failures;
  }

  // One rank failing must fail the whole test.
  int total_failures = 0;
  MPI_Allreduce(&g_failures, &total_failures, 1, MPI_INT, MPI_SUM,
                component_comm);
  if (g_rank == 0) {
    if (total_failures == 0) {
      std::printf("all checks passed on %d rank(s)\n", world_size);
    } else {
      std::printf("%d check(s) failed across %d rank(s)\n", total_failures,
                  world_size);
    }
  }

  MPI_Comm_free(&component_comm);
  MPI_Finalize();
  return total_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
