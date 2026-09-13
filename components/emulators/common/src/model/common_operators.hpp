/**
 * @file common_operators.hpp
 * @brief Operators any emulator can use: publishing to and averaging from
 *        the exchange, and insolation.
 */

#ifndef E3SM_EMULATOR_MODEL_COMMON_OPERATORS_HPP
#define E3SM_EMULATOR_MODEL_COMMON_OPERATORS_HPP

#include <string>
#include <utility>
#include <vector>

#include "operator.hpp"
#include "physics/insolation.hpp"

namespace emulator {
namespace model {

/**
 * @brief `exchange.publish`: copy fields to the exchange on every call.
 *
 * ```yaml
 * - operator: exchange.publish
 *   fields: {atm.TAUX: state.TAUX, ocn.sst: exports.So_t}
 *   clip_min_zero: [atm.surface_precipitation_rate]  # max(v, 0) on the copy
 * ```
 */
class ExchangePublish : public Operator {
public:
  ExchangePublish(const config::Section &options, const ModelInfo &info);
  void exports(const StepInfo &, Fields &f) override;

private:
  std::vector<std::pair<std::string, FieldRef>> m_fields;
  std::vector<bool> m_clip;
  std::vector<double> m_buffer;
};

/**
 * @brief A window mean of samples into forcing inputs, taken when the
 *        network steps: the forcing path of a long-step model.
 *
 * sample() adds one sample of every channel (derived classes say where it
 * comes from); before_step() writes each channel's mean into the input of
 * the same name, clips the named channels at zero, and copies each into
 * `<name><also_into_suffix>` when that is set; after_step() starts a new
 * window.  The sums are restart state.
 */
class WindowMeanForcing : public Operator {
public:
  WindowMeanForcing(const config::Section &options, const ModelInfo &info,
                    std::vector<std::string> channels);
  Declarations declarations() const override;
  void sample(const StepInfo &, Fields &f) override;
  void before_step(const StepInfo &, Fields &f) override;
  void after_step(const StepInfo &, Fields &f) override;
  void save_to(coupling::RestartStore &store,
               const std::string &prefix) const override;
  void load_from(coupling::RestartStore &store,
                 const std::string &prefix) override;

protected:
  /// Fill `sample`, which holds one field per channel.
  virtual void fill_sample(Fields &f, fields::FieldSet &sample) = 0;
  const std::vector<std::string> &channels() const { return m_channels; }

private:
  std::vector<std::string> m_channels;
  std::vector<std::string> m_clip_min_zero;
  std::string m_suffix;
  coupling::IntervalMean m_window;
  fields::FieldSet m_sample;
};

/**
 * @brief `exchange.window_mean`: forcing averaged from exchange fields.
 *
 * ```yaml
 * - operator: exchange.window_mean
 *   prefix: atm.                 # channel X is sampled from exchange atm.X
 *   channels: [TAUX, TAUY]
 *   also_into_suffix: ":next"
 *   clip_min_zero: [surface_precipitation_rate]
 * ```
 */
class ExchangeWindowMean : public WindowMeanForcing {
public:
  ExchangeWindowMean(const config::Section &options, const ModelInfo &info);

protected:
  void fill_sample(Fields &f, fields::FieldSet &sample) override;

private:
  std::string m_prefix;
};

/**
 * @brief `insolation`: top-of-atmosphere insolation for the network and the
 *        other operators.
 *
 * The input channel gets the mean over the step being predicted, moved
 * `offset_seconds` later (physics::Insolation::window_mean), at
 * initialization and before every network step.  `aux.solin_window` keeps
 * the mean over the step itself, which is what the instantaneous values the
 * coupler steps see average to, and `aux.solin_now` gets the instantaneous
 * value on every call.
 *
 * ```yaml
 * - operator: insolation
 *   channel: SOLIN
 *   orbit: {eccen: 0.016715, obliq: 23.4441, mvelp: 102.7}   # degrees
 *   solar_constant: 1360.53   # W/m2; default 1368.22 (RRTMG, EATM)
 *   offset_seconds: 1800      # default 0
 * ```
 *
 * SamudrACE's forcing SOLIN is matched to 2.0 W/m2 RMS by 1360.53 and 1800;
 * the defaults miss it by 52.6 W/m2 RMS and 1.9 W/m2 in the global mean.
 */
class InsolationOperator : public Operator {
public:
  InsolationOperator(const config::Section &options, const ModelInfo &info);
  Declarations declarations() const override;
  void initialize(const StepInfo &info, Fields &f) override;
  void before_step(const StepInfo &info, Fields &f) override;
  void exports(const StepInfo &info, Fields &f) override;

private:
  void window(const StepInfo &info, Fields &f);
  std::string m_channel;
  int m_offset = 0;
  physics::Insolation m_sun;
};

/// Registers the operators above.  Called by the registry itself.
void register_common_operators(OperatorRegistry &registry);

} // namespace model
} // namespace emulator

#endif // E3SM_EMULATOR_MODEL_COMMON_OPERATORS_HPP
