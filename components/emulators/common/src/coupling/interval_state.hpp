/**
 * @file interval_state.hpp
 * @brief What a long-step emulator carries across its interval: the mean of
 *        its forcing, and the two states the coupler is blended between.
 */

#ifndef E3SM_EMULATOR_COUPLING_INTERVAL_STATE_HPP
#define E3SM_EMULATOR_COUPLING_INTERVAL_STATE_HPP

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "field_set.hpp"
#include "restart_store.hpp"

namespace emulator {
namespace coupling {

/**
 * @brief Coupler fields averaged over an emulator interval.
 *
 * An emulator's flux channels are interval means -- `cell_methods: "time:
 * mean"` over its own step in the training data -- while the coupler
 * delivers a fresh value every coupling step.  Reading the coupler once at
 * the boundary hands a five-day-mean channel one half-hour sample.
 *
 * add() takes every channel from a FieldSet in one call, so a channel cannot
 * be left out of the loop -- the Fortran shape of this defect, where the sum
 * is short one step, the mean is off a few per cent, and nothing is NaN.
 */
class IntervalMean {
public:
  IntervalMean() = default;
  IntervalMean(std::vector<std::string> names, std::size_t npoints);

  /// Add one sample of every channel.
  /// @throws std::out_of_range if `fields` lacks a channel, naming it
  /// @throws std::invalid_argument if `fields` has a different point count
  void add(const fields::FieldSet &fields);

  int samples() const { return m_samples; }
  const std::vector<std::string> &names() const { return m_names; }

  /// @throws std::logic_error with no samples: an interval with no data has
  ///         no mean, and zeros would hand the model a dead ocean
  void mean(std::string_view name, std::span<double> out) const;

  /// Start a new interval.
  void reset();

  /// Sums and sample count, in the units they were added in.
  void save_to(RestartStore &store, std::string_view prefix) const;
  /**
   * @param missing Fatal, or StartEmpty for an accumulator a newer version
   *        added: an old restart does not have it, and zero samples with zero
   *        sums is a consistent state to start from.
   * @return false if it started empty because the restart lacked it
   */
  bool load_from(RestartStore &store, std::string_view prefix,
                 Missing missing = Missing::Fatal);

private:
  std::size_t index_of(std::string_view name) const;

  std::vector<std::string> m_names;
  std::size_t m_npoints = 0;
  std::vector<std::vector<double>> m_sums;
  int m_samples = 0;
};

/**
 * @brief The emulator states bracketing the current interval, and the blend.
 *
 * The emulator advances on its long step; the coupler runs much faster.
 * Handing the coupler each new prediction directly makes a five-day step in
 * sea surface temperature arrive at the atmosphere as a discontinuity.  So
 * the component keeps `lower` (valid at the start of the interval) and
 * `upper` (one emulator step later) and gives the coupler a linear blend.
 *
 * Only *snapshot* channels are interpolated.  A channel that is an interval
 * mean over the emulator step (a flux, precipitation, SOLIN) is already the
 * right value for every coupler step inside the interval, and interpolating
 * it lags it by half a step: in the Fortran ACE atmosphere, holding the mean
 * channels instead was worth 42 W/m2 at the surface.  So each channel has a
 * Temporal kind, and blend() holds the IntervalMean ones at the upper
 * bracket.
 *
 * Both brackets go to the restart.  Keeping only the newest restarts the
 * interpolation from a single state, and the coupler sees a different
 * surface for the rest of the interval.
 */
class BracketedState {
public:
  /// What a channel's value at an emulator step means in time.
  enum class Temporal {
    Snapshot,    ///< valid at that instant: interpolate between brackets
    IntervalMean ///< the mean over the step it ends: hold the upper bracket
  };

  BracketedState() = default;
  /// Every channel a snapshot.
  BracketedState(std::vector<std::string> names, std::size_t npoints);
  /// One Temporal per name.
  BracketedState(std::vector<std::string> names, std::vector<Temporal> kinds,
                 std::size_t npoints);

  /// Seed both brackets from one state: the initial condition, where there
  /// is nothing earlier to blend from.
  void set_both(const fields::FieldSet &state);

  /// Take a new prediction: upper becomes lower, `new_upper` becomes upper.
  /// @throws std::logic_error before set_both()
  void advance(const fields::FieldSet &new_upper);

  /**
   * @brief Write lower + f (upper - lower) into every snapshot channel of
   *        `out`, and upper into every interval-mean channel.
   *
   * With interpolation off, `out` gets the upper bracket whatever `f` is:
   * the ablation the Fortran ocean carried as `eocn_interp_state`.
   * @throws std::logic_error before set_both(); std::invalid_argument if f
   *         is outside [0, 1]
   */
  void blend(double f, fields::FieldSet &out) const;

  bool seeded() const { return m_seeded; }
  Temporal kind(std::string_view name) const { return m_kinds[index_of(name)]; }
  void set_interpolate(bool on) { m_interpolate = on; }
  bool interpolate() const { return m_interpolate; }

  std::span<const double> lower(std::string_view name) const;
  std::span<const double> upper(std::string_view name) const;

  /// Both brackets, every channel.  Missing either on load is fatal.
  void save_to(RestartStore &store, std::string_view prefix) const;
  void load_from(RestartStore &store, std::string_view prefix);

private:
  std::size_t index_of(std::string_view name) const;
  void copy_in(const fields::FieldSet &from, std::vector<std::vector<double>> &to,
               const char *what) const;

  std::vector<std::string> m_names;
  std::vector<Temporal> m_kinds;
  std::size_t m_npoints = 0;
  std::vector<std::vector<double>> m_lower;
  std::vector<std::vector<double>> m_upper;
  bool m_seeded = false;
  bool m_interpolate = true;
};

} // namespace coupling
} // namespace emulator

#endif // E3SM_EMULATOR_COUPLING_INTERVAL_STATE_HPP
