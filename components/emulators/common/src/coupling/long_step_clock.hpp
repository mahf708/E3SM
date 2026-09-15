/**
 * @file long_step_clock.hpp
 * @brief When an emulator whose step outlasts the coupler's should advance.
 */

#ifndef E3SM_EMULATOR_COUPLING_LONG_STEP_CLOCK_HPP
#define E3SM_EMULATOR_COUPLING_LONG_STEP_CLOCK_HPP

#include <cstdint>
#include <string>
#include <string_view>

#include "restart_store.hpp"

namespace emulator {
namespace coupling {

/// A model time as the driver reports it.  Only compared, never subtracted.
struct ModelTime {
  int ymd = -1; ///< yyyymmdd
  int tod = -1; ///< seconds into the day

  bool operator==(const ModelTime &) const = default;
  std::string to_string() const;
};

/**
 * @brief Counts coupler steps and says when the emulator advances.
 *
 * Samudra steps five days and the coupler half an hour; ACE steps six hours.
 * Two things go wrong when that cadence is derived rather than counted, and
 * the Fortran emulators hit both:
 *
 *  - calendar arithmetic (`mod` on the time of day) cannot express a step
 *    longer than a day at all;
 *  - the driver can call a component's run twice at one model time.  Advance
 *    twice there and the emulator is a step ahead for the rest of the run;
 *    merely *count* twice and every later advance comes early.
 *
 * So one call per coupler step, on_coupler_step(now), does all of it, and a
 * second call at the same `now` changes nothing and returns the same answer.
 *
 * The cadence must divide evenly -- 432000 s over 1800 s is 240 steps -- or
 * the emulator would drift against the calendar, so an uneven one is refused.
 */
class LongStepClock {
public:
  /// What to do on this coupler step.
  struct Step {
    /// False if on_coupler_step was already called at this model time.
    bool first_call = true;
    /// True on the coupler step that completes an emulator interval.
    bool advance = false;
    /// Where the coupler is in the current interval, in [0, 1): the blend
    /// weight between the bracketing states.  0 right after an advance.
    double fraction = 0.0;
    /// Emulator steps completed, counting this one if it advanced.  The
    /// restart-safe step index for InferenceBackend::set_step().
    std::int64_t completed_steps = 0;
  };

  LongStepClock() = default;

  /// @throws std::invalid_argument unless 0 < coupler_dt <= model_dt and
  ///         coupler_dt divides model_dt
  LongStepClock(int model_dt, int coupler_dt);

  Step on_coupler_step(ModelTime now);

  int model_dt() const { return m_model_dt; }
  int coupler_dt() const { return m_coupler_dt; }
  int steps_per_interval() const { return m_model_dt / m_coupler_dt; }
  /// Coupler steps counted into the current interval.
  int elapsed_steps() const { return m_elapsed_steps; }
  std::int64_t completed_steps() const { return m_completed_steps; }
  /// What the last on_coupler_step returned (restored by load_from).
  const Step &last_step() const { return m_last_step; }

  /// Every piece of state: elapsed steps, completed steps, the last model
  /// time (so the idempotence survives the restart too) and the cadence,
  /// which is checked on load.
  void save_to(RestartStore &store, std::string_view prefix) const;
  /// @throws std::runtime_error if anything is missing, or the restart was
  ///         written with a different cadence
  void load_from(RestartStore &store, std::string_view prefix);

  std::string to_string() const;

private:
  int m_model_dt = 0;
  int m_coupler_dt = 0;
  int m_elapsed_steps = 0;
  std::int64_t m_completed_steps = 0;
  ModelTime m_last_time;
  Step m_last_step;
};

} // namespace coupling
} // namespace emulator

#endif // E3SM_EMULATOR_COUPLING_LONG_STEP_CLOCK_HPP
