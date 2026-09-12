// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "network_stepper.hpp"

#include <mpi.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace emulator {
namespace coupling {
namespace test {

namespace {

constexpr int kNx = 4;
constexpr int kNy = 2;

/// FRAC coupled, PHIS boundary, SOLIN forcing, PS and T prognostic.
fields::ChannelLayout toy_layout() {
  fields::ChannelLayout l;
  l.name = "toy";
  l.model_dt = 21600;
  l.inputs = {"FRAC", "PHIS", "SOLIN", "PS", "T"};
  l.outputs = {"PS", "T", "FLUX"};
  l.coupled_inputs = {"FRAC"};
  l.boundary_inputs = {"PHIS"};
  l.forcing_inputs = {"SOLIN"};
  l.interval_mean_outputs = {"FLUX"};
  return l;
}

/**
 * A network with arithmetic a test can check, on the whole [1, C, ny, nx]
 * tensor: PS' = PS + 1, T' = T + SOLIN/100 + step, FLUX = PHIS + FRAC, and
 * FLUX[cell] = index of that cell, so a wrong cell order shows.  Step 99 puts
 * a NaN in FLUX.
 */
class ToyNetwork : public inference::InferenceBackend {
public:
  ToyNetwork()
      : InferenceBackend(inference::InferenceConfig{},
                         inference::InferenceContext{}) {}
  std::string name() const override { return "toy"; }
  std::vector<std::int64_t> seen_dims_in, seen_dims_out;

protected:
  void init_impl() override {}
  void final_impl() override {}
  bool infer_impl(const inference::TensorMap &inputs,
                  inference::TensorMap &outputs) override {
    const auto &in = *inputs.begin();
    auto &out = *outputs.begin();
    seen_dims_in = in.dims();
    seen_dims_out = out.dims();
    const std::size_t n = kNx * kNy;
    const double *x = in.cdata();
    double *y = out.data();
    auto X = [&](int c, std::size_t k) { return x[c * n + k]; };
    for (std::size_t k = 0; k < n; ++k) {
      y[0 * n + k] = X(3, k) + 1.0;
      y[1 * n + k] = X(4, k) + X(2, k) / 100.0 + static_cast<double>(step());
      y[2 * n + k] = X(1, k) + X(0, k) + 1000.0 * static_cast<double>(k);
    }
    if (step() == 99) {
      y[2 * n + 5] = std::numeric_limits<double>::quiet_NaN();
    }
    return true;
  }
};

struct Rank {
  int rank = 0, size = 1;
  Rank() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }
};

} // namespace

TEST_CASE("A whole-grid network steps every rank's columns", "[stepper]") {
  const Rank r;
  const auto decomp =
      grid::Decomposition::contiguous_blocks(kNx * kNy, r.size, r.rank);
  const grid::GlobalGather gather(MPI_COMM_WORLD, decomp);
  std::shared_ptr<ToyNetwork> net;
  if (gather.is_root()) {
    net = std::make_shared<ToyNetwork>();
    net->initialize();
  }
  NetworkStepper stepper(toy_layout(), MPI_COMM_WORLD, gather, kNx, kNy, net);

  // Initial condition and this step's coupled, boundary and forcing values,
  // each a function of the global cell so every rank can check its own.
  auto set = [&](const char *name, double base, double per_cell) {
    auto f = stepper.inputs().get(name);
    for (std::size_t i = 0; i < f.size(); ++i) {
      f[i] = base + per_cell * static_cast<double>(decomp.offset() + i);
    }
  };
  set("FRAC", 0.5, 0.0);
  set("PHIS", 10.0, 1.0);
  set("SOLIN", 300.0, 0.0);
  set("PS", 1e5, 0.0);
  set("T", 280.0, 0.0);

  stepper.step(7);
  if (net) {
    REQUIRE(net->seen_dims_in == std::vector<std::int64_t>{1, 5, kNy, kNx});
    REQUIRE(net->seen_dims_out == std::vector<std::int64_t>{1, 3, kNy, kNx});
  }
  const auto &p = stepper.prediction();
  for (std::size_t i = 0; i < decomp.num_local(); ++i) {
    const double cell = static_cast<double>(decomp.offset() + i);
    REQUIRE(p.get("PS")[i] == 1e5 + 1.0);
    REQUIRE(p.get("T")[i] == 280.0 + 3.0 + 7.0);
    REQUIRE(p.get("FLUX")[i] == 10.0 + cell + 0.5 + 1000.0 * cell);
    // Prognostic inputs now hold the prediction; the others are untouched.
    REQUIRE(stepper.inputs().get("PS")[i] == 1e5 + 1.0);
    REQUIRE(stepper.inputs().get("T")[i] == 290.0);
    REQUIRE(stepper.inputs().get("SOLIN")[i] == 300.0);
  }

  // A second step carries the state forward.
  stepper.step(8);
  REQUIRE(stepper.prediction().get("PS")[0] == 1e5 + 2.0);
  REQUIRE(stepper.prediction().get("T")[0] == 290.0 + 3.0 + 8.0);
}

TEST_CASE("A non-finite output stops every rank, with the channel named",
          "[stepper]") {
  const Rank r;
  const auto decomp =
      grid::Decomposition::contiguous_blocks(kNx * kNy, r.size, r.rank);
  const grid::GlobalGather gather(MPI_COMM_WORLD, decomp);
  std::shared_ptr<ToyNetwork> net;
  if (gather.is_root()) {
    net = std::make_shared<ToyNetwork>();
    net->initialize();
  }
  NetworkStepper stepper(toy_layout(), MPI_COMM_WORLD, gather, kNx, kNy, net);
  // Every rank, not just the root, must see the exception: otherwise the
  // rest would wait in the scatter forever.
  REQUIRE_THROWS_WITH(stepper.step(99), Catch::Contains("'FLUX'") &&
                                            Catch::Contains("cell 5") &&
                                            Catch::Contains("step 99"));
}

TEST_CASE("A tensor shape that is not the grid is refused", "[stepper]") {
  const Rank r;
  const auto decomp =
      grid::Decomposition::contiguous_blocks(kNx * kNy, r.size, r.rank);
  const grid::GlobalGather gather(MPI_COMM_WORLD, decomp);
  auto net = std::make_shared<ToyNetwork>();
  REQUIRE_THROWS_WITH(
      NetworkStepper(toy_layout(), MPI_COMM_WORLD, gather, 3, 3, net),
      Catch::Contains("3 x 3 tensor for a grid of 8"));
}

} // namespace test
} // namespace coupling
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
