/**
 * @file long_step_clock.cpp
 * @brief Implementation of LongStepClock.
 */

#include "long_step_clock.hpp"

#include <sstream>
#include <stdexcept>

namespace emulator {
namespace coupling {

std::string ModelTime::to_string() const {
  return std::to_string(ymd) + "-" + std::to_string(tod) + "s";
}

LongStepClock::LongStepClock(int model_dt, int coupler_dt)
    : m_model_dt(model_dt), m_coupler_dt(coupler_dt) {
  if (coupler_dt <= 0 || model_dt < coupler_dt) {
    throw std::invalid_argument(
        "A long-step clock needs 0 < coupler_dt <= model_dt; got model_dt " +
        std::to_string(model_dt) + " s and coupler_dt " +
        std::to_string(coupler_dt) + " s.");
  }
  if (model_dt % coupler_dt != 0) {
    throw std::invalid_argument(
        "The emulator step (" + std::to_string(model_dt) +
        " s) is not a whole number of coupler steps (" +
        std::to_string(coupler_dt) +
        " s), so its advances would drift against the calendar.");
  }
}

LongStepClock::Step LongStepClock::on_coupler_step(ModelTime now) {
  if (m_coupler_dt == 0) {
    throw std::logic_error("on_coupler_step on a default-constructed clock.");
  }
  if (now == m_last_time) {
    Step repeat = m_last_step;
    repeat.first_call = false;
    return repeat;
  }

  Step step;
  ++m_elapsed_steps;
  if (m_elapsed_steps == steps_per_interval()) {
    step.advance = true;
    m_elapsed_steps = 0;
    ++m_completed_steps;
  }
  step.fraction = static_cast<double>(m_elapsed_steps) /
                  static_cast<double>(steps_per_interval());
  step.completed_steps = m_completed_steps;

  m_last_time = now;
  m_last_step = step;
  return step;
}

void LongStepClock::save_to(RestartStore &store,
                            std::string_view prefix) const {
  store.write_int(restart_name(prefix, "model_dt"), m_model_dt);
  store.write_int(restart_name(prefix, "coupler_dt"), m_coupler_dt);
  store.write_int(restart_name(prefix, "elapsed_steps"), m_elapsed_steps);
  store.write_int(restart_name(prefix, "completed_steps"), m_completed_steps);
  store.write_int(restart_name(prefix, "last_ymd"), m_last_time.ymd);
  store.write_int(restart_name(prefix, "last_tod"), m_last_time.tod);
  store.write_int(restart_name(prefix, "last_advance"),
                  m_last_step.advance ? 1 : 0);
}

void LongStepClock::load_from(RestartStore &store, std::string_view prefix) {
  auto need = [&](const char *name) {
    std::int64_t value = 0;
    if (!store.read_int(restart_name(prefix, name), value)) {
      throw std::runtime_error("The restart has no '" +
                               restart_name(prefix, name) +
                               "'; the clock cannot be restored without it.");
    }
    return value;
  };

  const auto model_dt = need("model_dt");
  const auto coupler_dt = need("coupler_dt");
  if (model_dt != m_model_dt || coupler_dt != m_coupler_dt) {
    std::ostringstream oss;
    oss << "The restart's clock steps " << model_dt << " s over " << coupler_dt
        << " s coupler steps; this run is configured for " << m_model_dt
        << " s over " << m_coupler_dt
        << " s. Continuing would put the emulator's advances at different "
           "model times than the run that wrote the restart.";
    throw std::runtime_error(oss.str());
  }
  m_elapsed_steps = static_cast<int>(need("elapsed_steps"));
  m_completed_steps = need("completed_steps");
  m_last_time.ymd = static_cast<int>(need("last_ymd"));
  m_last_time.tod = static_cast<int>(need("last_tod"));

  m_last_step = Step{};
  m_last_step.advance = need("last_advance") != 0;
  m_last_step.fraction = static_cast<double>(m_elapsed_steps) /
                         static_cast<double>(steps_per_interval());
  m_last_step.completed_steps = m_completed_steps;
}

std::string LongStepClock::to_string() const {
  std::ostringstream oss;
  oss << "step " << m_model_dt << " s over " << m_coupler_dt
      << " s coupler steps; " << m_completed_steps << " completed, "
      << m_elapsed_steps << "/" << (m_coupler_dt ? steps_per_interval() : 0)
      << " into the next; last call at " << m_last_time.to_string();
  return oss.str();
}

} // namespace coupling
} // namespace emulator
