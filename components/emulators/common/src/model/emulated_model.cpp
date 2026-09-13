/**
 * @file emulated_model.cpp
 * @brief Implementation of ModelSpec and EmulatedModel.
 */

#include "emulated_model.hpp"

#include "channel_layout_yaml.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace emulator {
namespace model {

namespace {

bool contains(const std::vector<std::string> &names, const std::string &n) {
  return std::find(names.begin(), names.end(), n) != names.end();
}

std::vector<fields::FieldSpec> read_field_specs(const config::Section &s,
                                                const std::string &key) {
  std::vector<fields::FieldSpec> out;
  for (const auto &e : s.list(key)) {
    e.only({"name", "units", "need", "mask"});
    fields::FieldSpec f;
    f.name = e.string("name");
    f.units = e.string("units");
    const auto need = e.string_or("need", "required");
    if (need == "required") {
      f.need = fields::Need::Required;
    } else if (need == "optional") {
      f.need = fields::Need::Optional;
    } else {
      throw std::invalid_argument(e.where() + ".need: '" + need +
                                  "' is neither required nor optional.");
    }
    f.mask = e.string_or("mask", "");
    out.push_back(std::move(f));
  }
  return out;
}

} // namespace

ModelSpec ModelSpec::read(const config::Section &root) {
  ModelSpec s;
  s.name = root.string("name");
  if (root.has("network")) {
    s.layout = fields::read_channel_layout(root.section("network"));
    const auto stepping = root.string("stepping");
    if (stepping == "interpolate") {
      s.stepping = Stepping::Interpolate;
    } else if (stepping == "window_close") {
      s.stepping = Stepping::WindowClose;
    } else {
      throw std::invalid_argument(root.where() + "stepping: '" + stepping +
                                  "' is neither interpolate nor window_close.");
    }
  }
  const auto ic = root.optional_section("initial_condition");
  ic.only({"zero_fill", "computed", "strip_suffixes"});
  s.initial_condition.zero_fill = ic.names("zero_fill");
  s.initial_condition.computed = ic.names("computed");
  s.initial_condition.strip_suffixes = ic.names("strip_suffixes");
  s.operators = root.list("operators");
  const auto coupler = root.optional_section("coupler");
  coupler.only({"imports", "exports"});
  s.imports = read_field_specs(coupler, "imports");
  s.exports = read_field_specs(coupler, "exports");
  return s;
}

EmulatedModel::EmulatedModel(
    ModelSpec spec, int coupler_dt, Geometry geometry,
    std::shared_ptr<inference::InferenceBackend> backend,
    coupling::Exchange *exchange)
    : m_spec(std::move(spec)), m_coupler_dt(coupler_dt),
      m_geometry(std::move(geometry)), m_exchange(exchange),
      m_state(m_geometry.num_local()), m_aux(m_geometry.num_local()),
      m_statics(m_geometry.num_local()), m_no_inputs(m_geometry.num_local()) {
  const auto n = m_geometry.num_local();
  ModelInfo info;
  info.layout = has_network() ? &*m_spec.layout : nullptr;
  info.geometry = &m_geometry;
  info.coupler_dt = coupler_dt;
  const auto &registry = OperatorRegistry::instance();
  std::vector<std::string> written;
  for (const auto &entry : m_spec.operators) {
    m_operators.push_back(registry.create(entry, info));
    const auto d = m_operators.back()->declarations();
    for (const auto &a : d.aux) {
      if (!m_aux.contains(a)) {
        m_aux.add(a);
      }
    }
    for (const auto &s : d.statics) {
      if (!m_statics.contains(s)) {
        m_statics.add(s);
      }
    }
    written.insert(written.end(), d.writes_inputs.begin(),
                   d.writes_inputs.end());
  }

  // Every input the network does not carry forward or read from the
  // initial condition must be set by an operator: a channel nobody writes
  // would reach the network as zeros.
  if (has_network()) {
    const auto &layout = *m_spec.layout;
    for (const auto &in : layout.inputs) {
      const auto src = layout.source(in);
      if ((src == fields::InputSource::Coupled ||
           src == fields::InputSource::Forcing) &&
          !contains(written, in)) {
        throw std::invalid_argument(
            "EmulatedModel '" + m_spec.name + "': input '" + in + "' is " +
            fields::to_string(src) +
            " but no operator in the spec sets it.");
      }
    }
  }
  if (has_network()) {
    if (m_geometry.grid == nullptr) {
      throw std::invalid_argument("EmulatedModel '" + m_spec.name +
                                  "': a network needs the whole grid.");
    }
    const auto &layout = *m_spec.layout;
    m_stepper = std::make_unique<coupling::NetworkStepper>(
        layout, m_geometry.comm, *m_geometry.gather, m_geometry.grid->nx,
        m_geometry.grid->ny, std::move(backend));
    m_clock = coupling::LongStepClock(layout.model_dt, coupler_dt);
    m_brackets = coupling::BracketedState(layout.outputs,
                                          layout.output_temporals(), n);
    for (const auto &name : layout.outputs) {
      m_state.add(name);
    }
  }

}

std::vector<std::string> EmulatedModel::initial_condition_names() const {
  std::vector<std::string> names;
  auto add = [&](const std::string &n) {
    if (!contains(names, n)) {
      names.push_back(n);
    }
  };
  if (has_network()) {
    for (const auto &in : m_spec.layout->inputs) {
      if (contains(m_spec.initial_condition.computed, in)) {
        continue;
      }
      std::string base = in;
      for (const auto &suffix : m_spec.initial_condition.strip_suffixes) {
        if (base.size() > suffix.size() &&
            base.compare(base.size() - suffix.size(), suffix.size(), suffix) ==
                0) {
          base.resize(base.size() - suffix.size());
        }
      }
      add(base);
    }
  }
  for (const auto &s : m_statics.names()) {
    add(s);
  }
  return names;
}

void EmulatedModel::load(const std::vector<grid::GridField> &ic,
                         bool boundary_only) {
  auto find = [&](const std::string &name) -> const grid::GridField & {
    const auto it = std::find_if(
        ic.begin(), ic.end(), [&](const auto &f) { return f.name == name; });
    if (it == ic.end()) {
      throw std::runtime_error("The initial condition for '" + m_spec.name +
                               "' has no '" + name + "'.");
    }
    return *it;
  };
  const auto &policy = m_spec.initial_condition;
  const auto &decomp = m_geometry.decomp;
  if (has_network()) {
    const auto &layout = *m_spec.layout;
    auto &inputs = m_stepper->inputs();
    for (const auto &in : layout.inputs) {
      if (contains(policy.computed, in)) {
        continue;
      }
      if (boundary_only &&
          layout.source(in) != fields::InputSource::Boundary) {
        continue;
      }
      std::string base = in;
      for (const auto &suffix : policy.strip_suffixes) {
        if (base.size() > suffix.size() &&
            base.compare(base.size() - suffix.size(), suffix.size(), suffix) ==
                0) {
          base.resize(base.size() - suffix.size());
        }
      }
      const auto &field = find(base);
      auto values = field.values;
      if (field.unusable() > 0) {
        if (!contains(policy.zero_fill, base)) {
          throw std::runtime_error(
              "The initial condition's '" + base + "' has " +
              std::to_string(field.non_finite) + " non-finite and " +
              std::to_string(field.fill_like) +
              " fill-like values. Only initial_condition.zero_fill channels "
              "may be zero-filled; anything else would poison the network.");
        }
        for (auto &v : values) {
          if (!std::isfinite(v) || std::abs(v) >= 1e30) {
            v = 0.0;
          }
        }
      }
      const auto local = decomp.local(values);
      auto dest = inputs.get(in);
      std::copy(local.begin(), local.end(), dest.begin());
    }
  }
  for (const auto &name : m_statics.names()) {
    const auto local = decomp.local(find(name).values);
    auto dest = m_statics.get(name);
    std::copy(local.begin(), local.end(), dest.begin());
  }
}

Fields EmulatedModel::fields_for(const fields::FieldSet *imports,
                                 fields::FieldSet *exports) {
  Fields f;
  f.imports = imports;
  f.exports = exports;
  f.inputs = has_network() ? &m_stepper->inputs() : &m_no_inputs;
  f.state = &m_state;
  f.prediction = has_network() ? &m_stepper->prediction() : nullptr;
  f.upper = has_network() ? &m_brackets : nullptr;
  f.aux = &m_aux;
  f.statics = &m_statics;
  f.exchange = m_exchange;
  return f;
}

void EmulatedModel::initialize(coupling::ModelTime start,
                               const std::vector<grid::GridField> &ic) {
  load(ic, /*boundary_only=*/false);
  StepInfo info{start, {}, has_network() ? m_spec.layout->model_dt : 0};
  auto f = fields_for(nullptr, nullptr);
  for (auto &op : m_operators) {
    op->initialize(info, f);
  }
  if (has_network()) {
    const auto &layout = *m_spec.layout;
    auto &inputs = m_stepper->inputs();
    fields::FieldSet lower(m_geometry.num_local());
    for (const auto &name : layout.outputs) {
      auto dest = lower.add(name);
      if (inputs.contains(name)) {
        const auto from = inputs.get(name);
        std::copy(from.begin(), from.end(), dest.begin());
      }
    }
    if (m_spec.stepping == Stepping::Interpolate) {
      // The lower bracket is the initial state wherever an output has an
      // input of the same name, captured before the step overwrites the
      // prognostic inputs; outputs with no earlier value (the flux
      // channels) hold the first prediction in both brackets.
      m_stepper->step(0);
      for (const auto &name : layout.outputs) {
        if (!inputs.contains(name)) {
          const auto from = m_stepper->prediction().get(name);
          auto dest = lower.get(name);
          std::copy(from.begin(), from.end(), dest.begin());
        }
      }
      m_brackets.set_both(lower);
      m_brackets.advance(m_stepper->prediction());
    } else {
      m_brackets.set_both(lower);
    }
  }
  m_started = true;
}

void EmulatedModel::initial_exports(coupling::ModelTime start,
                                    const fields::FieldSet &imports,
                                    fields::FieldSet &exports) {
  if (!m_started) {
    throw std::logic_error("EmulatedModel '" + m_spec.name +
                           "': initial_exports before initialize.");
  }
  StepInfo info{start, {}, has_network() ? m_spec.layout->model_dt : 0};
  if (has_network()) {
    m_brackets.blend(m_spec.stepping == Stepping::Interpolate ? 0.0 : 1.0,
                     m_state);
  }
  auto f = fields_for(&imports, &exports);
  for (auto &op : m_operators) {
    op->exports(info, f);
  }
}

void EmulatedModel::run(coupling::ModelTime now,
                        const fields::FieldSet &imports,
                        fields::FieldSet &exports) {
  if (!m_started) {
    throw std::logic_error("EmulatedModel '" + m_spec.name +
                           "': run before initialize or restart.");
  }
  auto f = fields_for(&imports, &exports);
  StepInfo info{now, {}, 0};
  if (has_network()) {
    info.model_dt = m_spec.layout->model_dt;
    info.clock = m_clock.on_coupler_step(now);
    if (info.clock.first_call) {
      for (auto &op : m_operators) {
        op->sample(info, f);
      }
      if (info.clock.advance) {
        for (auto &op : m_operators) {
          op->before_step(info, f);
        }
        m_stepper->step(info.clock.completed_steps);
        m_brackets.advance(m_stepper->prediction());
        for (auto &op : m_operators) {
          op->after_step(info, f);
        }
      }
    }
    m_brackets.blend(m_spec.stepping == Stepping::Interpolate
                         ? info.clock.fraction
                         : 1.0,
                     m_state);
  }
  for (auto &op : m_operators) {
    op->exports(info, f);
  }
}

std::string EmulatedModel::restart_key(const std::string &what) const {
  return m_spec.name + "." + what;
}

void EmulatedModel::save_to(coupling::RestartStore &store) const {
  if (has_network()) {
    m_clock.save_to(store, restart_key("clock"));
    m_brackets.save_to(store, restart_key("state"));
  }
  // Operators share state through aux (the held insolation, say), so all of
  // it is restart state.
  for (const auto &name : m_aux.names()) {
    store.write_array(restart_key("aux." + name), m_aux.get(name));
  }
  for (std::size_t i = 0; i < m_operators.size(); ++i) {
    m_operators[i]->save_to(store, restart_key("operator" + std::to_string(i)));
  }
}

void EmulatedModel::restart(coupling::RestartStore &store,
                            const std::vector<grid::GridField> &ic) {
  load(ic, /*boundary_only=*/true);
  if (has_network()) {
    m_clock.load_from(store, restart_key("clock"));
    m_brackets.load_from(store, restart_key("state"));
  }
  for (const auto &name : m_aux.names()) {
    if (!store.read_array(restart_key("aux." + name), m_aux.get(name))) {
      throw std::runtime_error("The restart for '" + m_spec.name + "' has no '" +
                               restart_key("aux." + name) + "'.");
    }
  }
  for (std::size_t i = 0; i < m_operators.size(); ++i) {
    m_operators[i]->load_from(store, restart_key("operator" + std::to_string(i)));
  }
  if (has_network()) {
    // The network's next inputs are the upper bracket: the raw prediction.
    for (const auto &name :
         m_spec.layout->inputs_from(fields::InputSource::Prognostic)) {
      const auto from = m_brackets.upper(name);
      auto to = m_stepper->inputs().get(name);
      std::copy(from.begin(), from.end(), to.begin());
    }
  }
  m_started = true;
}

} // namespace model
} // namespace emulator
