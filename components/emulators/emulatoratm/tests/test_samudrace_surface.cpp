// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "ace_operators.hpp"
#include "emulated_model.hpp"
#include "emulator_test_support.hpp"
#include "samudrace_surface.hpp"

#include <cmath>
#include <limits>

namespace emulator {
namespace atm {
namespace test {

namespace {
const double nan = std::numeric_limits<double>::quiet_NaN();
}

TEST_CASE("The fractions and TS are fme's, on the same cells", "[samudrace]") {
  // fme's CoupledOceanFractionConfig.build_ocean_data, clip and per-name
  // masks, and Prescriber(TS, OCNFRAC, interpolate) on these seven cells
  // (ace/scratch/ocean_to_atm_ref/reference.py, float64):
  //            open  partial coastal land  ice  nan-ice coarse-land
  const std::vector<double> L{0.0, 0.3, 0.4, 1.0, 0.0, 0.1, 1.02};
  const std::vector<double> s{0.0, 0.5, 0.0, 0.0, 1.0, nan, 0.0};
  const std::vector<double> m2d{1, 1, 0, 0, 1, 1, 1};
  const std::vector<double> mice{1, 1, 0, 0, 1, 0, 1};
  const std::vector<double> sst{300.0, 271.5, 0.0, 0.0, 271.2, 290.0, 280.0};
  std::vector<double> ts{299.0, 260.0, 305.0, 250.0, 255.0, 289.0, 270.0};
  const std::vector<double> want_ocn{1, 0.34999999999999998, 0, 0, 0,
                                     0.90000000000000002, 0};
  const std::vector<double> want_ice{0, 0.34999999999999998, 0, 0, 1, 0, 0};
  const std::vector<double> want_ts{300, 264.02499999999998, 305, 250, 255,
                                    289.89999999999998, 270};
  std::vector<double> ocn(7), ice(7);
  samudrace_fractions({L, s, m2d, mice}, ocn, ice);
  samudrace_prescribe_ts(ocn, sst, m2d, ts);
  for (std::size_t i = 0; i < 7; ++i) {
    INFO("cell " << i);
    REQUIRE(ocn[i] == Approx(want_ocn[i]).margin(1e-15));
    REQUIRE(ice[i] == Approx(want_ice[i]).margin(1e-15));
    REQUIRE(ts[i] == Approx(want_ts[i]).margin(1e-12));
  }
}

namespace {

/// TS' = TS, so the carried value shows how often it was blended.
class Identity : public inference::InferenceBackend {
public:
  Identity()
      : InferenceBackend(inference::InferenceConfig{},
                         inference::InferenceContext{}) {}
  std::string name() const override { return "identity"; }

protected:
  void init_impl() override {}
  void final_impl() override {}
  bool infer_impl(const inference::TensorMap &inputs,
                  inference::TensorMap &outputs) override {
    const double *x = inputs.begin()->cdata();
    double *y = outputs.begin()->data();
    const std::size_t n = outputs.begin()->dims().back() *
                          outputs.begin()->dims()[2];
    // inputs: LANDFRAC, OCNFRAC, ICEFRAC, TS; outputs: TS
    for (std::size_t k = 0; k < n; ++k) {
      y[k] = x[3 * n + k];
    }
    return true;
  }
};

} // namespace

TEST_CASE("TS is blended once per step, and again only when the ocean "
          "changes", "[samudrace]") {
  register_atm_operators();
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  grid::HorizontalGrid g;
  g.name = "one";
  g.nx = 1;
  g.ny = 1;
  g.lat = {10.0};
  g.lon = {20.0};
  g.area = {4.0 * M_PI};
  if (size != 1) {
    WARN("one cell: runs on one rank only");
    return;
  }
  const auto decomp = grid::Decomposition::contiguous_blocks(1, 1, 0);
  const auto spec = model::ModelSpec::read(config::Section::load_string(R"(
name: toy
network:
  name: toy
  timestep: 3600
  inputs: [LANDFRAC, OCNFRAC, ICEFRAC, TS]
  outputs: [TS]
  coupled_inputs: [OCNFRAC, ICEFRAC, TS]
  boundary_inputs: [LANDFRAC]
stepping: interpolate
operators:
  - operator: samudrace.ocean_to_atmosphere
    land_fraction: inputs.LANDFRAC
    ocean: {sst: exchange.ocn.sst_raw, ice_fraction: exchange.ocn.sea_ice_fraction,
            ocean_mask: exchange.ocn.domain.mask, ice_mask: exchange.ocn.ice_mask}
    predicted_ts: upper.TS
    to: {ocnfrac: inputs.OCNFRAC, icefrac: inputs.ICEFRAC, ts: inputs.TS}
)", "toy.yaml"));
  auto net = std::make_shared<Identity>();
  net->initialize();
  coupling::Exchange ex;
  // A cell half land: OCNFRAC 0.5 with no ice.
  auto publish = [&](double sst) {
    ex.publish("ocn.sst_raw", std::vector<double>{sst});
    ex.publish("ocn.sea_ice_fraction", std::vector<double>{0.0});
    ex.publish("ocn.domain.mask", std::vector<double>{1.0});
    ex.publish("ocn.ice_mask", std::vector<double>{1.0});
  };
  publish(300.0);
  model::EmulatedModel m(spec, 1800,
                         model::Geometry::from_grid(MPI_COMM_WORLD, g, decomp),
                         net, &ex);
  std::vector<grid::GridField> ic(4);
  const char *names[] = {"LANDFRAC", "OCNFRAC", "ICEFRAC", "TS"};
  const double values[] = {0.5, 0.9, 0.0, 280.0};
  for (int k = 0; k < 4; ++k) {
    ic[k].name = names[k];
    ic[k].values = {values[k]};
  }
  m.initialize({20000101, 0}, ic);
  // Initialization prescribes the IC's TS once (290), the identity network
  // returns it, and after the step it is blended again (295): fme blends
  // each step's output.
  REQUIRE(m.aux().get("samudrace_ts")[0] == Approx(295.0));
  fields::FieldSet imports(1), exports(1);
  m.run({20000101, 1800}, imports, exports);
  m.run({20000101, 3600}, imports, exports); // step 1: same ocean
  // Input = carried 295 (no second blend: the ocean is unchanged); output
  // 295, blended once: 297.5.
  REQUIRE(m.aux().get("samudrace_ts")[0] == Approx(297.5));
  publish(310.0); // the ocean's window closes
  m.run({20000101, 5400}, imports, exports);
  m.run({20000101, 7200}, imports, exports); // step 2: new ocean
  // Input = 297.5 blended with 310 (303.75); output blended again: 306.875.
  REQUIRE(m.aux().get("samudrace_ts")[0] == Approx(306.875));
}

} // namespace test
} // namespace atm
} // namespace emulator

EMULATOR_TEST_MPI_MAIN
