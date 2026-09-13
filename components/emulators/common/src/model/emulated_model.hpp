/**
 * @file emulated_model.hpp
 * @brief A machine-learned model as a spec: a network, how it steps, and the
 *        operators around it.
 */

#ifndef E3SM_EMULATOR_MODEL_EMULATED_MODEL_HPP
#define E3SM_EMULATOR_MODEL_EMULATED_MODEL_HPP

#include <mpi.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "channel_layout.hpp"
#include "coupler_binding.hpp"
#include "grid_field_reader.hpp"
#include "inference_backend.hpp"
#include "interval_state.hpp"
#include "long_step_clock.hpp"
#include "model_fields.hpp"
#include "network_stepper.hpp"
#include "operator.hpp"
#include "restart_store.hpp"
#include "yaml_config.hpp"

namespace emulator {
namespace model {

/**
 * @brief How the network's steps line up with the coupler's.
 *
 * - Interpolate (ACE): the network steps at the start of each interval from
 *   the inputs then, predicting its end; the coupler sees snapshot outputs
 *   blended between the brackets and interval means held.  The first step is
 *   taken at initialization.
 * - WindowClose (Samudra in SamudrACE): the network steps when an interval
 *   closes, from the state at its start and forcing accumulated over it; the
 *   coupler sees the latest state, held.  No step at initialization.
 */
enum class Stepping { Interpolate, WindowClose };

/// How the initial condition fills the network's inputs.
struct InitialConditionPolicy {
  /// Channels whose NaN and fill values become 0 (coarsened fractions).
  std::vector<std::string> zero_fill;
  /// Inputs an operator computes, not read from the file.
  std::vector<std::string> computed;
  /// Suffixes stripped to find an input's variable ("TAUX:next" -> TAUX).
  std::vector<std::string> strip_suffixes;
};

/**
 * @brief Everything about a model that is not a file path.
 *
 * ```yaml
 * network:                  # fields::read_channel_layout; absent: no network
 * stepping: interpolate     # or window_close
 * initial_condition:
 *   zero_fill: [LANDFRAC]
 *   computed: [SOLIN]
 *   strip_suffixes: [":next"]
 * operators:                # in the order the model applies them
 *   - operator: insolation
 *     channel: SOLIN
 * coupler:
 *   imports: [{name: Sf_lfrac, units: "1"}]
 *   exports: [{name: Sa_z, units: m}, {name: Si_snowh, units: m, need: optional}]
 * ```
 */
struct ModelSpec {
  std::string name;
  std::optional<fields::ChannelLayout> layout;
  Stepping stepping = Stepping::Interpolate;
  InitialConditionPolicy initial_condition;
  std::vector<config::Section> operators;
  std::vector<fields::FieldSpec> imports;
  std::vector<fields::FieldSpec> exports;

  /// @throws std::invalid_argument naming the key
  static ModelSpec read(const config::Section &root);
};

class EmulatedModel {
public:
  /**
   * @param backend  the network, on the root rank; null elsewhere, and
   *                 ignored without a network
   * @param exchange in-process fields shared with other components; may be
   *                 null if no operator uses it
   * @throws std::invalid_argument if the spec has a network and the geometry
   *         no grid, or an input no operator sets
   */
  EmulatedModel(ModelSpec spec, int coupler_dt, Geometry geometry,
                std::shared_ptr<inference::InferenceBackend> backend,
                coupling::Exchange *exchange);

  EmulatedModel(const EmulatedModel &) = delete;
  EmulatedModel &operator=(const EmulatedModel &) = delete;

  /// Variables initialize() reads from the initial condition file.
  std::vector<std::string> initial_condition_names() const;

  /// Load the initial condition; with Interpolate, take the first step.
  void initialize(coupling::ModelTime start,
                  const std::vector<grid::GridField> &initial_condition);
  void initial_exports(coupling::ModelTime start,
                       const fields::FieldSet &imports,
                       fields::FieldSet &exports);
  void run(coupling::ModelTime now, const fields::FieldSet &imports,
           fields::FieldSet &exports);

  void save_to(coupling::RestartStore &store) const;
  /// Restore from a restart, reloading boundary inputs and statics from the
  /// initial condition file.
  void restart(coupling::RestartStore &store,
               const std::vector<grid::GridField> &initial_condition);

  const ModelSpec &spec() const { return m_spec; }
  bool has_network() const { return m_spec.layout.has_value(); }
  const coupling::LongStepClock &clock() const { return m_clock; }
  const coupling::BracketedState &brackets() const { return m_brackets; }
  const fields::FieldSet &aux() const { return m_aux; }
  const fields::FieldSet &statics() const { return m_statics; }
  const fields::FieldSet &state() const { return m_state; }
  const Geometry &geometry() const { return m_geometry; }

private:
  Fields fields_for(const fields::FieldSet *imports, fields::FieldSet *exports);
  void load(const std::vector<grid::GridField> &ic, bool boundary_only);
  std::string restart_key(const std::string &what) const;

  ModelSpec m_spec;
  int m_coupler_dt;
  Geometry m_geometry;
  coupling::Exchange *m_exchange;
  std::vector<std::unique_ptr<Operator>> m_operators;
  std::unique_ptr<coupling::NetworkStepper> m_stepper;
  coupling::LongStepClock m_clock;
  coupling::BracketedState m_brackets;
  fields::FieldSet m_state;
  fields::FieldSet m_aux;
  fields::FieldSet m_statics;
  fields::FieldSet m_no_inputs;
  bool m_started = false;
};

} // namespace model
} // namespace emulator

#endif // E3SM_EMULATOR_MODEL_EMULATED_MODEL_HPP
