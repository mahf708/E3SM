// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "emulator_test_support.hpp"
#include "global_gather.hpp"
#include "horizontal_grid.hpp"
#include "restart_file.hpp"

#include <mpi.h>

#include <unistd.h>

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace emulator {
namespace test {

namespace {

using coupling::MemoryRestartStore;
using coupling::read_restart_file;
using coupling::write_restart_file;

int rank_of(MPI_Comm comm) {
  int r = 0;
  MPI_Comm_rank(comm, &r);
  return r;
}

/// One path on every rank, in the working directory (shared across nodes).
std::string shared_path(const std::string &stem) {
  int pid = static_cast<int>(::getpid());
  MPI_Bcast(&pid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return stem + "_" + std::to_string(pid) + ".nc";
}

/// Throws on every rank, or on none.
bool throws_everywhere(const std::function<void()> &body) {
  int threw = 0;
  try {
    body();
  } catch (const std::runtime_error &) {
    threw = 1;
  }
  int lo = 0, hi = 0;
  MPI_Allreduce(&threw, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&threw, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(lo == hi);
  return lo == 1;
}

} // namespace

TEST_CASE("A restart file written on any number of ranks reads back on any "
          "other", "[restart]") {
  if (!coupling::have_restart_files()) {
    WARN("skipped: this build has no netCDF");
    return;
  }
  // 11 cells: uneven blocks on any rank count above one.
  const std::size_t ncells = 11;
  const int size = [] {
    int s = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
  }();
  const int rank = rank_of(MPI_COMM_WORLD);
  const auto decomp = grid::Decomposition::contiguous_blocks(ncells, size, rank);
  const grid::GlobalGather gather(MPI_COMM_WORLD, decomp);

  MemoryRestartStore store;
  std::vector<double> a(decomp.num_local()), b(decomp.num_local());
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto c = static_cast<double>(decomp.offset() + i);
    a[i] = 0.1 * c + 1.0 / 3.0; // not representable in decimal: bitwise
    b[i] = -c;
  }
  store.write_array("ocean.state.sst:next", a);
  store.write_array("ocean.aux.b", b);
  store.write_int("ocean.clock.completed_steps", 123456789012LL);
  store.write_int("ocean.clock.last_tod", -1);

  const auto path = shared_path("restart_file_test");
  write_restart_file(path, store, gather, MPI_COMM_WORLD,
                     {{"component", "emulatoratm"}});

  SECTION("on the ranks that wrote it") {
    auto back = read_restart_file(path, gather, MPI_COMM_WORLD);
    std::vector<double> got(decomp.num_local());
    REQUIRE(back.read_array("ocean.state.sst:next", got));
    REQUIRE(got == a);
    REQUIRE(back.read_array("ocean.aux.b", got));
    REQUIRE(got == b);
    std::int64_t v = 0;
    REQUIRE(back.read_int("ocean.clock.completed_steps", v));
    REQUIRE(v == 123456789012LL);
    REQUIRE(back.read_int("ocean.clock.last_tod", v));
    REQUIRE(v == -1);
    REQUIRE(back.arrays().size() == 2);
    REQUIRE(back.ints().size() == 2);
  }

  SECTION("on one rank, with the whole grid") {
    // Every rank reads the file alone, as a one-rank run would.
    const auto whole = grid::Decomposition::contiguous_blocks(ncells, 1, 0);
    const grid::GlobalGather one(MPI_COMM_SELF, whole);
    auto back = read_restart_file(path, one, MPI_COMM_SELF);
    std::vector<double> got(ncells);
    REQUIRE(back.read_array("ocean.state.sst:next", got));
    for (std::size_t c = 0; c < ncells; ++c) {
      REQUIRE(got[c] == 0.1 * static_cast<double>(c) + 1.0 / 3.0);
    }
  }

  SECTION("a grid of another size is refused on every rank") {
    const auto other = grid::Decomposition::contiguous_blocks(ncells + 1, size,
                                                              rank);
    const grid::GlobalGather wrong(MPI_COMM_WORLD, other);
    REQUIRE(throws_everywhere(
        [&] { read_restart_file(path, wrong, MPI_COMM_WORLD); }));
  }

  MPI_Barrier(MPI_COMM_WORLD); // every rank done with the file
  if (rank == 0) {
    std::remove(path.c_str());
  }
}

TEST_CASE("A restart that cannot be written or read fails on every rank",
          "[restart]") {
  if (!coupling::have_restart_files()) {
    WARN("skipped: this build has no netCDF");
    return;
  }
  const int size = [] {
    int s = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
  }();
  const int rank = rank_of(MPI_COMM_WORLD);
  const auto decomp = grid::Decomposition::contiguous_blocks(5, size, rank);
  const grid::GlobalGather gather(MPI_COMM_WORLD, decomp);

  SECTION("a missing file") {
    REQUIRE(throws_everywhere([&] {
      read_restart_file("no_such_dir/restart.nc", gather, MPI_COMM_WORLD);
    }));
  }
  SECTION("a directory that does not exist") {
    MemoryRestartStore store;
    store.write_int("x", 1);
    REQUIRE(throws_everywhere([&] {
      write_restart_file("no_such_dir/restart.nc", store, gather,
                         MPI_COMM_WORLD, {});
    }));
  }
  SECTION("an array that is not one value per cell") {
    MemoryRestartStore store;
    store.write_array("short", std::vector<double>(decomp.num_local() + 1));
    REQUIRE(throws_everywhere([&] {
      write_restart_file(shared_path("never_written"), store, gather,
                         MPI_COMM_WORLD, {});
    }));
  }
  SECTION("ranks that disagree on an integer") {
    MemoryRestartStore store;
    store.write_int("clock", rank);
    if (size > 1) {
      REQUIRE(throws_everywhere([&] {
        write_restart_file(shared_path("never_written"), store, gather,
                           MPI_COMM_WORLD, {});
      }));
    }
  }
}

} // namespace test
} // namespace emulator

EMULATOR_TEST_MPI_MAIN
