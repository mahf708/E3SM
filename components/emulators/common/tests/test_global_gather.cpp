// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "global_gather.hpp"
#include "horizontal_grid.hpp"

#include <mpi.h>

#include <vector>

namespace emulator {
namespace grid {
namespace test {

namespace {

int rank_of(MPI_Comm comm) {
  int r = 0;
  MPI_Comm_rank(comm, &r);
  return r;
}
int size_of(MPI_Comm comm) {
  int s = 0;
  MPI_Comm_size(comm, &s);
  return s;
}

} // namespace

TEST_CASE("Gather assembles the global field in cell order, and scatter "
          "undoes it", "[gather]") {
  // 7 cells: uneven blocks on any rank count above one.
  const std::size_t ncells = 7;
  const int size = size_of(MPI_COMM_WORLD);
  const int rank = rank_of(MPI_COMM_WORLD);
  const auto decomp = Decomposition::contiguous_blocks(ncells, size, rank);
  const GlobalGather gather(MPI_COMM_WORLD, decomp);

  // Each rank's local values are the global cell indices it owns, times 10.
  std::vector<double> local(decomp.num_local());
  for (std::size_t i = 0; i < local.size(); ++i) {
    local[i] = 10.0 * static_cast<double>(decomp.offset() + i);
  }

  std::vector<double> global(gather.is_root() ? ncells : 0, -1.0);
  gather.gather(local, global);
  if (gather.is_root()) {
    for (std::size_t c = 0; c < ncells; ++c) {
      REQUIRE(global[c] == 10.0 * static_cast<double>(c));
    }
    // What a model would do on the root: change the whole field.
    for (auto &v : global) {
      v += 0.5;
    }
  }

  std::vector<double> back(decomp.num_local(), -1.0);
  gather.scatter(global, back);
  for (std::size_t i = 0; i < back.size(); ++i) {
    REQUIRE(back[i] == local[i] + 0.5);
  }
}

TEST_CASE("Gather refuses blocks that do not tile the grid", "[gather]") {
  const int size = size_of(MPI_COMM_WORLD);
  const int rank = rank_of(MPI_COMM_WORLD);
  // Every rank claims a block of a 9-cell grid as if there were one more
  // rank than there is: the blocks stop short of the end.
  const auto short_blocks = Decomposition::contiguous_blocks(9, size + 1, rank);
  REQUIRE_THROWS_WITH(GlobalGather(MPI_COMM_WORLD, short_blocks),
                      Catch::Contains("cells of a 9-cell grid"));
}

} // namespace test
} // namespace grid
} // namespace emulator

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Catch::Session session;
  if (rank != 0) {
    session.configData().outputFilename = "%debug";
  }
  int status = session.run(argc, argv);
  int worst = 0;
  MPI_Allreduce(&status, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return worst;
}
