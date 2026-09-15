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
 * A long-step emulator's interval can exceed a day, so calendar arithmetic
 * (`mod` on time of day) cannot derive its cadence, and the driver may call
 * a component's run twice at one model time. This counts coupler steps
 * instead: on_coupler_step(now) is the single entry point, and calling it
 * again at the same `now` is a no-op that returns the previous answer. The
 * cadence must divide evenly into the emulator step, or the emulator would
 * drift against the calendar.
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
