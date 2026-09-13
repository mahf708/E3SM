/**
 * @file history.hpp
 * @brief Time means of a model's fields, written on the whole grid.
 */

#ifndef E3SM_EMULATOR_MODEL_HISTORY_HPP
#define E3SM_EMULATOR_MODEL_HISTORY_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "long_step_clock.hpp"
#include "model_fields.hpp"
#include "restart_store.hpp"
#include "yaml_config.hpp"

namespace emulator {
namespace model {

/**
 * @brief History output: the mean of named fields over each interval.
 *
 * The coupler's own history is a snapshot, and its ocean fields are a step
 * old, so neither says what an emulator did over a month.  This does.  The
 * input file's section:
 *
 * ```yaml
 * history:
 *   prefix: mycase.emulatoratm.h   # file: <prefix>.YYYY-MM-DD-SSSSS.nc
 *   interval: monthly              # monthly, <n>d or <n>h
 *   fields: [state.TS, state.PS, exports.Sa_tbot, aux.samudrace_ts]
 * ```
 *
 * Each interval's file holds the mean of the fields at the end of every
 * coupler step in it, on (lat, lon), stamped with the interval's end; cells
 * outside the component's domain are _FillValue.  A `<n>d`/`<n>h` interval
 * counts from the run's start, so `5d` lines up with the coupler's history
 * every 5 days; `monthly` closes at 00:00 on the first of each month, so the
 * first and last months of a run can be partial (the `samples` attribute
 * says how many steps each file averages).  A driver call repeated at one
 * time is counted once.  The partial means are restart state.
 */
class History {
public:
  /// @throws std::invalid_argument naming the key on a bad interval or field
  History(const config::Section &options, const Geometry &geometry, int nx,
          int ny, int coupler_dt, std::string component);

  /// After the model's run at `now`.  Collective when an interval closes.
  void after_step(coupling::ModelTime now, const Fields &f);

  void save_to(coupling::RestartStore &store) const;
  /// Restores a partial interval; returns false (and starts afresh) if the
  /// restart has none for these fields.
  bool load_from(coupling::RestartStore &store);

  const std::string &prefix() const { return m_prefix; }
  std::int64_t samples() const { return m_samples; }
  /// The files written so far by this process.
  const std::vector<std::string> &written() const { return m_written; }

private:
  bool closes_interval(coupling::ModelTime now) const;
  void write(coupling::ModelTime end);

  const Geometry *m_geometry;
  int m_nx, m_ny;
  int m_coupler_dt;
  std::string m_component;
  std::string m_prefix;
  std::string m_interval;
  int m_interval_steps = 0; ///< 0 for monthly
  std::vector<FieldRef> m_fields;
  std::vector<std::string> m_names; ///< variable names in the file
  std::vector<std::vector<double>> m_sums;
  std::int64_t m_samples = 0;
  std::int64_t m_steps = 0; ///< coupler steps since the run started
  coupling::ModelTime m_start;
  coupling::ModelTime m_last;
  std::vector<std::string> m_written;
};

} // namespace model
} // namespace emulator

#endif // E3SM_EMULATOR_MODEL_HISTORY_HPP
