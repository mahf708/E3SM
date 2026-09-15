/**
 * @file channel_layout.hpp
 * @brief The channels a network reads and writes, and where each input
 *        comes from.
 */

#ifndef E3SM_EMULATOR_FIELDS_CHANNEL_LAYOUT_HPP
#define E3SM_EMULATOR_FIELDS_CHANNEL_LAYOUT_HPP

#include <string>
#include <string_view>
#include <vector>

#include "interval_state.hpp"

namespace emulator {
namespace fields {

/**
 * @brief Where an input channel's value comes from at each step.
 *
 * Every input has exactly one source, and a layout that leaves one without
 * a source is refused when it is validated, not discovered as a channel of
 * zeros inside the network.
 */
enum class InputSource {
  Prognostic, ///< carried forward from the output of the same name
  Coupled,    ///< set by the component from the coupler, even if predicted
  Boundary,   ///< fixed from the initial condition (or restart)
  Forcing     ///< computed by the component (e.g. insolation)
};

const char *to_string(InputSource source);

/**
 * @brief One network's channel contract, as data.
 *
 * The order of `inputs` and `outputs` is the tensor's channel order and is
 * fixed by the checkpoint.  Which outputs are interval means is a property of
 * the training data (`cell_methods`), not a modelling choice, and decides how
 * the component carries each to the coupler (coupling::BracketedState).
 */
struct ChannelLayout {
  std::string name;
  int model_dt = 0; ///< seconds

  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  /// Inputs the component sets, overriding any output of the same name.
  std::vector<std::string> coupled_inputs;
  std::vector<std::string> boundary_inputs;
  std::vector<std::string> forcing_inputs;

  /// Outputs that are means over the step they end; the rest are snapshots.
  std::vector<std::string> interval_mean_outputs;

  /**
   * @throws std::invalid_argument naming the channel, if names repeat, a
   *         role or mean names a channel the layout does not have, an input
   *         has two roles, or an input has no source at all
   */
  void validate() const;

  InputSource source(std::string_view input) const;
  coupling::BracketedState::Temporal temporal(std::string_view output) const;

  /// Inputs with this source, in input order.
  std::vector<std::string> inputs_from(InputSource source) const;
  /// Temporal kind of every output, in output order.
  std::vector<coupling::BracketedState::Temporal> output_temporals() const;
};

} // namespace fields
} // namespace emulator

#endif // E3SM_EMULATOR_FIELDS_CHANNEL_LAYOUT_HPP
