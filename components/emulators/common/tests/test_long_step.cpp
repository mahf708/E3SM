// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "exchange.hpp"
#include "field_set.hpp"
#include "interval_state.hpp"
#include "long_step_clock.hpp"
#include "restart_store.hpp"

#include <cmath>
#include <vector>

namespace emulator {
namespace coupling {
namespace test {

namespace {

/// Model time after `n` half-hour coupler steps from 2000-01-01 00:00.
ModelTime after_steps(int n) {
  const int seconds = n * 1800;
  return {20000101 + seconds / 86400, seconds % 86400};
}

} // namespace

// ---------------------------------------------------------------------------
// LongStepClock
// ---------------------------------------------------------------------------

TEST_CASE("A five-day emulator advances once in 240 half-hour steps",
          "[long_step][clock]") {
  LongStepClock clock(5 * 86400, 1800);
  REQUIRE(clock.steps_per_interval() == 240);

  int advances = 0;
  for (int n = 1; n <= 240; ++n) {
    const auto step = clock.on_coupler_step(after_steps(n));
    REQUIRE(step.first_call);
    if (n < 240) {
      REQUIRE_FALSE(step.advance);
      REQUIRE(step.fraction == Approx(n / 240.0));
    } else {
      REQUIRE(step.advance);
      REQUIRE(step.fraction == 0.0);
    }
    advances += step.advance ? 1 : 0;
  }
  REQUIRE(advances == 1);
  REQUIRE(clock.completed_steps() == 1);
}

TEST_CASE("A second run at the same model time changes nothing",
          "[long_step][clock]") {
  LongStepClock clock(6 * 3600, 1800); // ACE: 12 coupler steps

  for (int n = 1; n <= 11; ++n) {
    clock.on_coupler_step(after_steps(n));
    // The driver calls run again at the same time, every time.
    const auto again = clock.on_coupler_step(after_steps(n));
    REQUIRE_FALSE(again.first_call);
    REQUIRE_FALSE(again.advance);
  }
  // Counting the repeats would have advanced at call 6 of 22.
  REQUIRE(clock.elapsed_steps() == 11);
  REQUIRE(clock.completed_steps() == 0);

  const auto boundary = clock.on_coupler_step(after_steps(12));
  REQUIRE(boundary.advance);
  const auto repeat = clock.on_coupler_step(after_steps(12));
  REQUIRE(repeat.advance); // the same answer, reported again...
  REQUIRE_FALSE(repeat.first_call); // ...and marked, so no second advance
  REQUIRE(clock.completed_steps() == 1);
}

TEST_CASE("A cadence that would drift is refused", "[long_step][clock]") {
  REQUIRE_THROWS_WITH(LongStepClock(86400, 7 * 60),
                      Catch::Contains("not a whole number"));
  REQUIRE_THROWS_AS(LongStepClock(1800, 3600), std::invalid_argument);
  REQUIRE_THROWS_AS(LongStepClock(1800, 0), std::invalid_argument);

  LongStepClock every_step(1800, 1800);
  REQUIRE(every_step.on_coupler_step(after_steps(1)).advance);
  REQUIRE(every_step.on_coupler_step(after_steps(2)).advance);
}

TEST_CASE("A restarted clock continues exactly, repeats included",
          "[long_step][clock][restart]") {
  LongStepClock continuous(6 * 3600, 1800);
  std::vector<LongStepClock::Step> reference;
  for (int n = 1; n <= 100; ++n) {
    reference.push_back(continuous.on_coupler_step(after_steps(n)));
  }

  LongStepClock first_half(6 * 3600, 1800);
  for (int n = 1; n <= 40; ++n) {
    first_half.on_coupler_step(after_steps(n));
  }
  MemoryRestartStore store;
  first_half.save_to(store, "ocn.clock");

  LongStepClock second_half(6 * 3600, 1800);
  second_half.load_from(store, "ocn.clock");
  // The restarted driver's first call is at the time the restart was
  // written: a repeat, and it must not count.
  REQUIRE_FALSE(second_half.on_coupler_step(after_steps(40)).first_call);
  for (int n = 41; n <= 100; ++n) {
    const auto step = second_half.on_coupler_step(after_steps(n));
    const auto &ref = reference[static_cast<std::size_t>(n - 1)];
    REQUIRE(step.advance == ref.advance);
    REQUIRE(step.fraction == ref.fraction);
    REQUIRE(step.completed_steps == ref.completed_steps);
  }
}

TEST_CASE("A clock refuses a restart with another cadence or missing state",
          "[long_step][clock][restart]") {
  LongStepClock ace(6 * 3600, 1800);
  ace.on_coupler_step(after_steps(1));
  MemoryRestartStore store;
  ace.save_to(store, "clock");

  LongStepClock samudra(5 * 86400, 1800);
  REQUIRE_THROWS_WITH(samudra.load_from(store, "clock"),
                      Catch::Contains("21600 s over 1800") &&
                          Catch::Contains("432000 s over 1800"));

  store.erase("clock.last_tod");
  LongStepClock again(6 * 3600, 1800);
  REQUIRE_THROWS_WITH(again.load_from(store, "clock"),
                      Catch::Contains("clock.last_tod"));
}

// ---------------------------------------------------------------------------
// IntervalMean
// ---------------------------------------------------------------------------

TEST_CASE("An interval mean averages every channel over its samples",
          "[long_step][mean]") {
  fields::FieldSet imports(2);
  auto swnet = imports.add("Foxx_swnet");
  auto taux = imports.add("Foxx_taux");

  IntervalMean mean({"Foxx_swnet", "Foxx_taux"}, 2);
  for (int k = 0; k < 4; ++k) {
    swnet[0] = 100.0 * k; // 0, 100, 200, 300: mean 150
    swnet[1] = 50.0;
    taux[0] = (k % 2 == 0) ? 0.1 : -0.1; // mean 0
    taux[1] = 0.2;
    mean.add(imports);
  }
  REQUIRE(mean.samples() == 4);

  std::vector<double> out(2);
  mean.mean("Foxx_swnet", out);
  REQUIRE(out == std::vector<double>{150.0, 50.0});
  mean.mean("Foxx_taux", out);
  REQUIRE(out[0] == Approx(0.0).margin(1e-15));
  REQUIRE(out[1] == Approx(0.2));

  mean.reset();
  REQUIRE(mean.samples() == 0);
  REQUIRE_THROWS_WITH(mean.mean("Foxx_swnet", out),
                      Catch::Contains("no samples"));
}

TEST_CASE("An interval mean refuses a sample missing a channel, untouched",
          "[long_step][mean]") {
  fields::FieldSet imports(1);
  imports.add("Foxx_swnet")[0] = 10.0;

  IntervalMean mean({"Foxx_swnet", "Foxx_lat"}, 1);
  REQUIRE_THROWS_WITH(mean.add(imports), Catch::Contains("'Foxx_lat'"));
  REQUIRE(mean.samples() == 0);

  imports.add("Foxx_lat")[0] = -5.0;
  mean.add(imports);
  std::vector<double> out(1);
  mean.mean("Foxx_swnet", out);
  REQUIRE(out[0] == 10.0); // not 20: the failed add left no trace
}

TEST_CASE("An interval mean survives a restart mid-interval",
          "[long_step][mean][restart]") {
  fields::FieldSet imports(3);
  auto f = imports.add("Faxa_rain");
  IntervalMean continuous({"Faxa_rain"}, 3);
  IntervalMean before({"Faxa_rain"}, 3);
  for (int k = 1; k <= 7; ++k) {
    for (std::size_t p = 0; p < 3; ++p) {
      f[p] = 1e-5 * k * (p + 1);
    }
    continuous.add(imports);
    if (k <= 4) {
      before.add(imports);
    }
  }
  MemoryRestartStore store;
  before.save_to(store, "acc");

  IntervalMean after({"Faxa_rain"}, 3);
  REQUIRE(after.load_from(store, "acc"));
  for (int k = 5; k <= 7; ++k) {
    for (std::size_t p = 0; p < 3; ++p) {
      f[p] = 1e-5 * k * (p + 1);
    }
    after.add(imports);
  }
  std::vector<double> a(3), b(3);
  continuous.mean("Faxa_rain", a);
  after.mean("Faxa_rain", b);
  REQUIRE(a == b); // bit for bit

  SECTION("an accumulator a newer version added starts empty, if allowed") {
    MemoryRestartStore old_restart;
    IntervalMean added({"Faxa_rain"}, 3);
    REQUIRE_THROWS_AS(added.load_from(old_restart, "acc"), std::runtime_error);
    REQUIRE_FALSE(added.load_from(old_restart, "acc", Missing::StartEmpty));
    REQUIRE(added.samples() == 0);
  }

  SECTION("a count without its sums is not a state to start from") {
    store.erase("acc.sum.Faxa_rain");
    IntervalMean partial({"Faxa_rain"}, 3);
    REQUIRE_THROWS_WITH(partial.load_from(store, "acc", Missing::StartEmpty),
                        Catch::Contains("acc.sum.Faxa_rain"));
  }
}

// ---------------------------------------------------------------------------
// BracketedState
// ---------------------------------------------------------------------------

TEST_CASE("A bracketed state blends between its brackets",
          "[long_step][bracket]") {
  fields::FieldSet state(2);
  auto sst = state.add("So_t");
  sst[0] = 280.0;
  sst[1] = 300.0;

  BracketedState brackets({"So_t"}, 2);
  REQUIRE_THROWS_AS(brackets.advance(state), std::logic_error);
  brackets.set_both(state);

  sst[0] = 290.0;
  sst[1] = 300.0;
  brackets.advance(state);
  REQUIRE(brackets.lower("So_t")[0] == 280.0);
  REQUIRE(brackets.upper("So_t")[0] == 290.0);

  fields::FieldSet out(2);
  out.add("So_t");
  brackets.blend(0.0, out);
  REQUIRE(out.get("So_t")[0] == 280.0);
  brackets.blend(0.25, out);
  REQUIRE(out.get("So_t")[0] == 282.5);
  REQUIRE(out.get("So_t")[1] == 300.0);

  brackets.set_interpolate(false);
  brackets.blend(0.25, out);
  REQUIRE(out.get("So_t")[0] == 290.0);

  REQUIRE_THROWS_AS(brackets.blend(1.5, out), std::invalid_argument);
}

TEST_CASE("Interval-mean channels are held, snapshots interpolated",
          "[long_step][bracket]") {
  using Temporal = BracketedState::Temporal;
  fields::FieldSet state(1);
  auto ts = state.add("TS");
  auto lh = state.add("LHFLX");
  BracketedState brackets({"TS", "LHFLX"},
                          {Temporal::Snapshot, Temporal::IntervalMean}, 1);
  ts[0] = 280.0;
  lh[0] = 80.0;
  brackets.set_both(state);
  ts[0] = 284.0;
  lh[0] = 120.0;
  brackets.advance(state);

  fields::FieldSet out(1);
  out.add("TS");
  out.add("LHFLX");
  brackets.blend(0.25, out);
  REQUIRE(out.get("TS")[0] == 281.0);
  // The 6 h mean ending at the upper bracket is the flux for every coupler
  // step in that window; blending would lag it by half a step.
  REQUIRE(out.get("LHFLX")[0] == 120.0);
  REQUIRE(brackets.kind("LHFLX") == Temporal::IntervalMean);

  REQUIRE_THROWS_AS(BracketedState({"TS", "LHFLX"}, {Temporal::Snapshot}, 1),
                    std::invalid_argument);
}

TEST_CASE("A bracketed state needs both brackets back from a restart",
          "[long_step][bracket][restart]") {
  fields::FieldSet state(1);
  auto sst = state.add("So_t");
  BracketedState brackets({"So_t"}, 1);
  sst[0] = 280.0;
  brackets.set_both(state);
  sst[0] = 290.0;
  brackets.advance(state);

  MemoryRestartStore store;
  brackets.save_to(store, "ocn.state");
  BracketedState restored({"So_t"}, 1);
  restored.load_from(store, "ocn.state");
  REQUIRE(restored.lower("So_t")[0] == 280.0);
  REQUIRE(restored.upper("So_t")[0] == 290.0);

  store.erase("ocn.state.lower.So_t");
  BracketedState half({"So_t"}, 1);
  REQUIRE_THROWS_WITH(half.load_from(store, "ocn.state"),
                      Catch::Contains("Both brackets"));
}

// ---------------------------------------------------------------------------
// All of it together
// ---------------------------------------------------------------------------

namespace {

/**
 * A toy long-step component, driven the way an MCT cap drives one: every
 * coupler step it accumulates its forcing, and on the step that closes an
 * interval it "runs the model" and advances its brackets; every step it
 * exports the blend.  The model is deterministic in (state, mean forcing,
 * step index) -- the step index standing in for the per-step reseed of a
 * stochastic emulator.
 */
class ToyOcean {
public:
  ToyOcean()
      : clock(6 * 3600, 1800), forcing({"Foxx_swnet"}, 4),
        brackets({"So_t"}, 4), imports(4), prediction(4), exports(4) {
    imports.add("Foxx_swnet");
    prediction.add("So_t");
    exports.add("So_t");
    auto sst = prediction.get("So_t");
    for (std::size_t p = 0; p < 4; ++p) {
      sst[p] = 280.0 + p;
    }
    brackets.set_both(prediction);
  }

  void run(ModelTime now, int n) {
    const auto step = clock.on_coupler_step(now);
    if (!step.first_call) {
      return;
    }
    auto flux = imports.get("Foxx_swnet");
    for (std::size_t p = 0; p < 4; ++p) {
      flux[p] = 100.0 + 10.0 * std::sin(0.1 * n + p);
    }
    forcing.add(imports);
    if (step.advance) {
      std::vector<double> mean(4);
      forcing.mean("Foxx_swnet", mean);
      auto next = prediction.get("So_t");
      const auto upper = brackets.upper("So_t");
      const double noise = 1e-3 * std::cos(1.7 * step.completed_steps);
      for (std::size_t p = 0; p < 4; ++p) {
        next[p] = upper[p] + 1e-3 * (mean[p] - 100.0) + noise;
      }
      brackets.advance(prediction);
      forcing.reset();
    }
    brackets.blend(step.fraction, exports);
  }

  void save(RestartStore &store) const {
    clock.save_to(store, "clock");
    forcing.save_to(store, "forcing");
    brackets.save_to(store, "state");
  }
  void load(RestartStore &store) {
    clock.load_from(store, "clock");
    forcing.load_from(store, "forcing");
    brackets.load_from(store, "state");
  }

  LongStepClock clock;
  IntervalMean forcing;
  BracketedState brackets;
  fields::FieldSet imports;
  fields::FieldSet prediction;
  fields::FieldSet exports;
};

} // namespace

TEST_CASE("A long-step component restarted mid-interval exports the same "
          "surface, bit for bit, through repeated driver calls",
          "[long_step][restart]") {
  const int total = 203; // 16 intervals and change, restart inside the 4th
  const int restart_at = 43;

  std::vector<std::vector<double>> continuous;
  ToyOcean a;
  for (int n = 1; n <= total; ++n) {
    a.run(after_steps(n), n);
    if (n % 17 == 0) {
      a.run(after_steps(n), n); // the driver's occasional second call
    }
    const auto sst = a.exports.get("So_t");
    continuous.emplace_back(sst.begin(), sst.end());
  }
  REQUIRE(a.clock.completed_steps() == total / 12);

  ToyOcean b;
  for (int n = 1; n <= restart_at; ++n) {
    b.run(after_steps(n), n);
    if (n % 17 == 0) {
      b.run(after_steps(n), n);
    }
  }
  MemoryRestartStore store;
  b.save(store);

  ToyOcean c;
  c.load(store);
  c.run(after_steps(restart_at), restart_at); // the restarted driver repeats
  for (int n = restart_at + 1; n <= total; ++n) {
    c.run(after_steps(n), n);
    if (n % 17 == 0) {
      c.run(after_steps(n), n);
    }
    const auto sst = c.exports.get("So_t");
    const auto &ref = continuous[static_cast<std::size_t>(n - 1)];
    for (std::size_t p = 0; p < 4; ++p) {
      INFO("step " << n << " point " << p);
      REQUIRE(sst[p] == ref[p]);
    }
  }
}

TEST_CASE("The exchange hands fields between components and guards the size",
          "[exchange]") {
  Exchange ex;
  REQUIRE_FALSE(ex.has("atm.FLDS"));
  REQUIRE(ex.publishes("atm.FLDS") == 0);
  REQUIRE_THROWS_WITH(ex.get("atm.FLDS"),
                      Catch::Contains("Nothing has published 'atm.FLDS'"));

  const std::vector<double> flds{300.0, 310.0};
  ex.publish("atm.FLDS", flds);
  REQUIRE(ex.get("atm.FLDS")[1] == 310.0);
  REQUIRE(ex.publishes("atm.FLDS") == 1);

  const std::vector<double> later{305.0, 315.0};
  ex.publish("atm.FLDS", later);
  REQUIRE(ex.get("atm.FLDS")[0] == 305.0); // a copy, updated in place
  REQUIRE(ex.publishes("atm.FLDS") == 2);

  const std::vector<double> other_grid{1.0, 2.0, 3.0};
  REQUIRE_THROWS_WITH(ex.publish("atm.FLDS", other_grid),
                      Catch::Contains("share the grid decomposition"));
}

} // namespace test
} // namespace coupling
} // namespace emulator
