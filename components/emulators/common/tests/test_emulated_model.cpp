// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "common_operators.hpp"
#include "emulated_model.hpp"
#include "grid_field_reader.hpp"
#include "history.hpp"
#include "restart_file.hpp"

#include <mpi.h>

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <memory>
#include <vector>

namespace emulator {
namespace model {
namespace test {

namespace {

constexpr int kNx = 4;
constexpr int kNy = 2;

/// A regular 4 x 2 grid on the sphere (the model only needs lat, lon, area).
grid::HorizontalGrid toy_grid() {
  grid::HorizontalGrid g;
  g.name = "toy";
  g.nx = kNx;
  g.ny = kNy;
  for (int j = 0; j < kNy; ++j) {
    for (int i = 0; i < kNx; ++i) {
      g.lat.push_back(j == 0 ? -45.0 : 45.0);
      g.lon.push_back(45.0 + 90.0 * i);
      g.area.push_back(4.0 * M_PI / (kNx * kNy));
    }
  }
  return g;
}

/**
 * Channels in layout order; outputs are X' = X + F + step and FLUX = 10 *
 * step + cell index, so each value says which inputs and step made it.
 */
class ToyNetwork : public inference::InferenceBackend {
public:
  ToyNetwork()
      : InferenceBackend(inference::InferenceConfig{},
                         inference::InferenceContext{}) {}
  std::string name() const override { return "toy"; }
  int calls = 0;

protected:
  void init_impl() override {}
  void final_impl() override {}
  bool infer_impl(const inference::TensorMap &inputs,
                  inference::TensorMap &outputs) override {
    ++calls;
    const std::size_t n = kNx * kNy;
    const double *x = inputs.begin()->cdata();
    double *y = outputs.begin()->data();
    for (std::size_t k = 0; k < n; ++k) {
      y[k] = x[n + k] + x[k] + static_cast<double>(step()); // X
      y[n + k] = 10.0 * static_cast<double>(step()) + static_cast<double>(k);
    }
    return true;
  }
};

/// `test.set`: writes a constant into an input before each step.
class SetOperator : public Operator {
public:
  SetOperator(const config::Section &o, const ModelInfo &)
      : m_channel(o.string("channel")), m_value(o.number("value")) {}
  Declarations declarations() const override {
    Declarations d;
    d.writes_inputs = {m_channel};
    return d;
  }
  void initialize(const StepInfo &, Fields &f) override { set(f); }
  void before_step(const StepInfo &, Fields &f) override { set(f); }

private:
  void set(Fields &f) {
    for (auto &v : f.inputs->get(m_channel)) {
      v = m_value;
    }
  }
  std::string m_channel;
  double m_value;
};

void register_test_operators() {
  OperatorRegistry::instance().add(
      "test.set", [](const config::Section &o, const ModelInfo &i) {
        return std::make_unique<SetOperator>(o, i);
      });
}

struct World {
  int rank = 0, size = 1;
  World() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }
};

coupling::ModelTime after(int n, int dt) {
  const int s = n * dt;
  return {20000101 + s / 86400, s % 86400};
}

std::vector<grid::GridField> toy_ic() {
  grid::GridField x;
  x.name = "X";
  x.values.assign(kNx * kNy, 0.0);
  for (int k = 0; k < kNx * kNy; ++k) {
    x.values[k] = static_cast<double>(k);
  }
  grid::GridField f;
  f.name = "F";
  f.values.assign(kNx * kNy, 0.0);
  return {x, f};
}

const char *kInterpolate = R"(
name: toy
network:
  name: toy
  timestep: 7200
  inputs: [F, X]
  outputs: [X, FLUX]
  interval_mean_outputs: [FLUX]
  forcing_inputs: [F]
stepping: interpolate
initial_condition: {computed: [F]}
operators:
  - {operator: test.set, channel: F, value: 1.0}
  - operator: exchange.publish
    fields: {toy.X: state.X, toy.FLUX: state.FLUX}
    clip_min_zero: [toy.X]
coupler:
  exports: [{name: So_x, units: "1"}]
)";

} // namespace

TEST_CASE("Interpolate: the first step at initialization, snapshots blended "
          "and means held between steps", "[model]") {
  register_test_operators();
  World w;
  const auto g = toy_grid();
  const auto decomp = grid::Decomposition::contiguous_blocks(g.size(), w.size, w.rank);
  auto net = std::make_shared<ToyNetwork>();
  net->initialize();
  coupling::Exchange exchange;
  const auto spec =
      ModelSpec::read(config::Section::load_string(kInterpolate, "toy.yaml"));
  REQUIRE(spec.exports.size() == 1);
  EmulatedModel model(spec, 1800, Geometry::from_grid(MPI_COMM_WORLD, g, decomp),
                      w.rank == 0 ? net : nullptr, &exchange);
  REQUIRE(model.initial_condition_names() == std::vector<std::string>{"X"});

  model.initialize({20000101, 0}, toy_ic());
  fields::FieldSet imports(decomp.num_local()), exports(decomp.num_local());
  model.initial_exports({20000101, 0}, imports, exports);
  const auto off = decomp.offset();
  for (std::size_t i = 0; i < decomp.num_local(); ++i) {
    const double k = static_cast<double>(off + i);
    // Blend 0: the initial X; FLUX held at the first prediction (step 0).
    REQUIRE(model.state().get("X")[i] == k);
    REQUIRE(model.state().get("FLUX")[i] == k);
    REQUIRE(exchange.get("toy.X")[i] == k);
  }
  // Half way to the first advance: X blended from k to k + 1 (F = 1, step 0).
  // The clock counts coupler steps, so every one is called.
  model.run(after(1, 1800), imports, exports);
  model.run(after(2, 1800), imports, exports);
  for (std::size_t i = 0; i < decomp.num_local(); ++i) {
    const double k = static_cast<double>(off + i);
    REQUIRE(model.state().get("X")[i] == Approx(k + 0.5));
    REQUIRE(model.state().get("FLUX")[i] == k);
  }
  model.run(after(3, 1800), imports, exports);
  model.run(after(4, 1800), imports, exports); // advance: step 1
  model.run(after(4, 1800), imports, exports); // a repeat changes nothing
  REQUIRE(model.clock().completed_steps() == 1);
  if (w.rank == 0) {
    REQUIRE(net->calls == 2);
  }
  for (std::size_t i = 0; i < decomp.num_local(); ++i) {
    const double k = static_cast<double>(off + i);
    REQUIRE(model.state().get("X")[i] == k + 1.0); // lower bracket
    REQUIRE(model.state().get("FLUX")[i] == 10.0 + k);
  }
}

TEST_CASE("WindowClose: forcing averaged over the window, one step at its "
          "close, the state held", "[model]") {
  World w;
  const auto g = toy_grid();
  const auto decomp = grid::Decomposition::contiguous_blocks(g.size(), w.size, w.rank);
  const auto n = decomp.num_local();
  auto net = std::make_shared<ToyNetwork>();
  net->initialize();
  coupling::Exchange exchange;
  const auto spec = ModelSpec::read(config::Section::load_string(R"(
name: toy
network:
  name: toy
  timestep: 7200
  inputs: [F, X]
  outputs: [X, FLUX]
  forcing_inputs: [F]
stepping: window_close
operators:
  - operator: exchange.window_mean
    prefix: other.
    channels: [F]
    clip_min_zero: [F]
)", "toy.yaml"));
  EmulatedModel model(spec, 1800, Geometry::from_grid(MPI_COMM_WORLD, g, decomp),
                      w.rank == 0 ? net : nullptr, &exchange);
  model.initialize({20000101, 0}, toy_ic());
  if (w.rank == 0) {
    REQUIRE(net->calls == 0); // no step at initialization
  }
  fields::FieldSet imports(n), exports(n);
  // Samples -3, 1, 2, 4 at steps 1..4: mean 1.  A window of -3 alone would
  // clip to 0.
  const double samples[] = {-3.0, 1.0, 2.0, 4.0};
  for (int s = 1; s <= 4; ++s) {
    exchange.publish("other.F", std::vector<double>(n, samples[s - 1]));
    model.run(after(s, 1800), imports, exports);
    if (s < 4) {
      for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(model.state().get("X")[i] == static_cast<double>(decomp.offset() + i));
      }
    }
  }
  REQUIRE(model.clock().completed_steps() == 1);
  for (std::size_t i = 0; i < n; ++i) {
    const double k = static_cast<double>(decomp.offset() + i);
    // X' = X + mean F + step 1, held.
    REQUIRE(model.state().get("X")[i] == k + 1.0 + 1.0);
  }
}

TEST_CASE("A spec is refused when an input has no source or an operator is "
          "unknown", "[model]") {
  World w;
  const auto g = toy_grid();
  const auto decomp = grid::Decomposition::contiguous_blocks(g.size(), w.size, w.rank);
  const auto no_setter = ModelSpec::read(config::Section::load_string(R"(
name: toy
network: {name: toy, timestep: 7200, inputs: [F, X], outputs: [X], forcing_inputs: [F]}
stepping: interpolate
)", "a.yaml"));
  REQUIRE_THROWS_WITH(
      EmulatedModel(no_setter, 1800, Geometry::from_grid(MPI_COMM_WORLD, g, decomp),
                    nullptr, nullptr),
      Catch::Contains("'F' is forcing but no operator"));
  const auto unknown = ModelSpec::read(config::Section::load_string(R"(
name: toy
operators: [{operator: no.such.thing}]
)", "b.yaml"));
  REQUIRE_THROWS_WITH(
      EmulatedModel(unknown, 1800, Geometry::from_grid(MPI_COMM_WORLD, g, decomp),
                    nullptr, nullptr),
      Catch::Contains("b.yaml: operators[0]") &&
          Catch::Contains("no operator 'no.such.thing'") &&
          Catch::Contains("exchange.publish"));
  REQUIRE_THROWS_WITH(
      FieldRef::parse("stat.X", "c.yaml: x"),
      Catch::Contains("unknown set 'stat'"));
}

TEST_CASE("A restart mid-interval continues exactly", "[model]") {
  World w;
  const auto g = toy_grid();
  const auto decomp = grid::Decomposition::contiguous_blocks(g.size(), w.size, w.rank);
  const auto n = decomp.num_local();
  const auto spec =
      ModelSpec::read(config::Section::load_string(kInterpolate, "toy.yaml"));
  auto run = [&](int from, int to, EmulatedModel &m,
                 std::vector<std::vector<double>> &seen) {
    fields::FieldSet imports(n), exports(n);
    for (int s = from; s <= to; ++s) {
      m.run(after(s, 1800), imports, exports);
      std::vector<double> v(m.state().get("X").begin(), m.state().get("X").end());
      v.insert(v.end(), m.state().get("FLUX").begin(), m.state().get("FLUX").end());
      seen.push_back(v);
    }
  };
  coupling::Exchange ex1, ex2, ex3;
  auto net1 = std::make_shared<ToyNetwork>();
  auto net2 = std::make_shared<ToyNetwork>();
  net1->initialize();
  net2->initialize();
  EmulatedModel whole(spec, 1800, Geometry::from_grid(MPI_COMM_WORLD, g, decomp),
                      w.rank == 0 ? net1 : nullptr, &ex1);
  whole.initialize({20000101, 0}, toy_ic());
  std::vector<std::vector<double>> continuous, first, second;
  run(1, 6, whole, continuous);
  // What the continuous run published at the step the restart is taken.
  const auto published = ex1.get("toy.X");
  const std::vector<double> at_restart(published.begin(), published.end());
  run(7, 13, whole, continuous);

  EmulatedModel a(spec, 1800, Geometry::from_grid(MPI_COMM_WORLD, g, decomp),
                  w.rank == 0 ? net2 : nullptr, &ex2);
  a.initialize({20000101, 0}, toy_ic());
  run(1, 6, a, first);
  coupling::MemoryRestartStore store;
  a.save_to(store);
  // A new process: its own exchange, which the restart's initial exports
  // must fill with what the continuous run held there, mid-interval (6 of
  // 4 coupler steps per network step: halfway).
  EmulatedModel b(spec, 1800, Geometry::from_grid(MPI_COMM_WORLD, g, decomp),
                  w.rank == 0 ? net2 : nullptr, &ex3);
  b.restart(store, toy_ic());
  REQUIRE(b.clock().last_step().fraction == 0.5);
  {
    fields::FieldSet imports(n), exports(n);
    b.initial_exports({20000101, 6 * 1800}, imports, exports);
  }
  const auto republished = ex3.get("toy.X");
  REQUIRE(std::vector<double>(republished.begin(), republished.end()) ==
          at_restart);
  run(7, 13, b, second);
  first.insert(first.end(), second.begin(), second.end());
  REQUIRE(first == continuous);
}

TEST_CASE("History writes each interval's mean on the grid, through repeated "
          "calls and a restart", "[model][history]") {
  if (!coupling::have_restart_files()) {
    WARN("skipped: this build has no netCDF");
    return;
  }
  register_test_operators();
  World w;
  const auto g = toy_grid();
  const auto decomp =
      grid::Decomposition::contiguous_blocks(g.size(), w.size, w.rank);
  const auto n = decomp.num_local();
  const auto spec =
      ModelSpec::read(config::Section::load_string(kInterpolate, "toy.yaml"));
  int pid = static_cast<int>(::getpid());
  MPI_Bcast(&pid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  const std::string tag = "history_test_" + std::to_string(pid);
  auto options = [&](const std::string &prefix, const std::string &fields) {
    return config::Section::load_string(
        "interval: 2h\nprefix: " + prefix + "\nfields: " + fields + "\n",
        "history");
  };
  auto net = std::make_shared<ToyNetwork>();
  net->initialize();
  auto make = [&](coupling::Exchange &ex) {
    return std::make_unique<EmulatedModel>(
        spec, 1800, Geometry::from_grid(MPI_COMM_WORLD, g, decomp),
        w.rank == 0 ? net : nullptr, &ex);
  };
  fields::FieldSet imports(n), exports(n);

  {
    const auto geometry = Geometry::from_grid(MPI_COMM_WORLD, g, decomp);
    // 2 h is not a whole number of 1700 s steps; "fortnightly" is no interval.
    REQUIRE_THROWS(History(options(tag, "[state.X]"), geometry, kNx, kNy, 1700,
                           "toy"));
    REQUIRE_THROWS(History(config::Section::load_string(
                               "interval: fortnightly\nfields: [state.X]\n",
                               "history"),
                           geometry, kNx, kNy, 1800, "toy"));
  }

  // Continuous: two 2-hour intervals of 4 steps, every call repeated.
  const std::string a = tag + "_a", b = tag + "_b";
  std::vector<double> mean_x(n, 0.0);
  {
    coupling::Exchange ex;
    auto m = make(ex);
    m->initialize({20000101, 0}, toy_ic());
    History h(options(a, "[state.X, upper.X]"), m->geometry(), kNx, kNy, 1800,
              "toy");
    for (int s = 1; s <= 8; ++s) {
      m->run(after(s, 1800), imports, exports);
      h.after_step(after(s, 1800), m->view(imports, exports));
      h.after_step(after(s, 1800), m->view(imports, exports));
      for (std::size_t i = 0; s > 4 && i < n; ++i) {
        mean_x[i] += m->state().get("X")[i];
      }
    }
    REQUIRE(h.written() == std::vector<std::string>{
                               a + ".2000-01-01-07200.nc",
                               a + ".2000-01-01-14400.nc"});
    REQUIRE(h.samples() == 0);
  }
  // Restarted at step 6, halfway through the second interval.
  {
    coupling::Exchange ex1, ex2;
    auto m1 = make(ex1);
    m1->initialize({20000101, 0}, toy_ic());
    History h1(options(b, "[state.X, upper.X]"), m1->geometry(), kNx, kNy,
               1800, "toy");
    for (int s = 1; s <= 6; ++s) {
      m1->run(after(s, 1800), imports, exports);
      h1.after_step(after(s, 1800), m1->view(imports, exports));
    }
    coupling::MemoryRestartStore store;
    m1->save_to(store);
    h1.save_to(store);
    auto m2 = make(ex2);
    m2->restart(store, toy_ic());
    History h2(options(b, "[state.X, upper.X]"), m2->geometry(), kNx, kNy,
               1800, "toy");
    REQUIRE(h2.load_from(store));
    REQUIRE(h2.samples() == 2);
    for (int s = 7; s <= 8; ++s) {
      m2->run(after(s, 1800), imports, exports);
      h2.after_step(after(s, 1800), m2->view(imports, exports));
    }
    REQUIRE(h2.written().size() == 1);
    // A history whose fields the restart does not hold starts afresh.
    History other(options(b, "[upper.X]"), m2->geometry(), kNx, kNy, 1800,
                  "toy");
    coupling::MemoryRestartStore empty;
    REQUIRE_FALSE(other.load_from(empty));
  }

  const auto file_a = grid::read_grid_fields(a + ".2000-01-01-14400.nc",
                                             {"state_X", "upper_X"}, kNy, kNx);
  const auto file_b = grid::read_grid_fields(b + ".2000-01-01-14400.nc",
                                             {"state_X", "upper_X"}, kNy, kNx);
  REQUIRE(file_a[0].values == file_b[0].values);
  REQUIRE(file_a[1].values == file_b[1].values);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(file_a[0].values[decomp.offset() + i] ==
            Approx(mean_x[i] / 4.0).epsilon(1e-14));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (w.rank == 0) {
    for (const auto &p : {a, b}) {
      std::remove((p + ".2000-01-01-07200.nc").c_str());
      std::remove((p + ".2000-01-01-14400.nc").c_str());
    }
  }
}

} // namespace test
} // namespace model
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
