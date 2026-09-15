/**
 * @file operator.hpp
 * @brief A named, tested piece of numerics a spec applies to fields.
 */

#ifndef E3SM_EMULATOR_MODEL_OPERATOR_HPP
#define E3SM_EMULATOR_MODEL_OPERATOR_HPP

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "channel_layout.hpp"
#include "coupler_binding.hpp"
#include "model_fields.hpp"
#include "restart_store.hpp"
#include "yaml_config.hpp"

namespace emulator {
namespace model {

/// What an operator needs the model to provide.
struct Declarations {
  std::vector<std::string> statics; ///< names read from the initial condition
  std::vector<std::string> aux;     ///< arrays it creates, this rank's size
  std::vector<std::string> writes_inputs; ///< input channels it sets
};

/// What an operator is built with.
struct ModelInfo {
  const fields::ChannelLayout *layout = nullptr; ///< null without a network
  const Geometry *geometry = nullptr;
  int coupler_dt = 0;
};

/**
 * @brief One step of numerics, applied where the model's spec says.
 *
 * The model calls, per coupler step, in spec order:
 *  - sample():      on the first call at a model time (window accumulation)
 *  - before_step(): when the network is about to step (sets inputs)
 *  - after_step():  after it stepped
 *  - exports():     every call, and for the initial exports
 * and initialize() once, after the initial condition is loaded and before
 * any network step (an interpolating model takes its first step then, and
 * calls after_step() right after).  Physics belongs in plain, tested
 * functions; an operator just connects one to fields named in the spec.
 */
class Operator {
public:
  virtual ~Operator() = default;
  virtual Declarations declarations() const { return {}; }
  virtual void initialize(const StepInfo &, Fields &) {}
  virtual void sample(const StepInfo &, Fields &) {}
  virtual void before_step(const StepInfo &, Fields &) {}
  virtual void after_step(const StepInfo &, Fields &) {}
  virtual void exports(const StepInfo &, Fields &) {}
  virtual void save_to(coupling::RestartStore &, const std::string &) const {}
  virtual void load_from(coupling::RestartStore &, const std::string &) {}
};

/// Builds an operator from its spec entry (the whole map, `operator` key
/// included).
using OperatorFactory = std::function<std::unique_ptr<Operator>(
    const config::Section &options, const ModelInfo &info)>;

/**
 * @brief Operator names to factories.
 *
 * Generic operators register with the registry itself; each component
 * library registers its own (register_atm_operators, ...) before it creates
 * a model, since static registration does not survive static linking.
 */
class OperatorRegistry {
public:
  static OperatorRegistry &instance();
  /// Replaces an earlier factory of the same name.
  void add(const std::string &name, OperatorFactory factory);
  bool has(const std::string &name) const;
  /// @throws std::invalid_argument naming the entry and the known operators
  std::unique_ptr<Operator> create(const config::Section &entry,
                                   const ModelInfo &info) const;
  std::vector<std::string> names() const;

private:
  OperatorRegistry();
  std::map<std::string, OperatorFactory> m_factories;
};

} // namespace model
} // namespace emulator

#endif // E3SM_EMULATOR_MODEL_OPERATOR_HPP
