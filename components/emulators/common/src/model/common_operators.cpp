/**
 * @file common_operators.cpp
 * @brief Implementation of the common operators and the registry.
 */

#include "common_operators.hpp"

#include <algorithm>
#include <stdexcept>

namespace emulator {
namespace model {

// ---------------------------------------------------------------------------
// OperatorRegistry
// ---------------------------------------------------------------------------

OperatorRegistry::OperatorRegistry() { register_common_operators(*this); }

OperatorRegistry &OperatorRegistry::instance() {
  static OperatorRegistry registry;
  return registry;
}

void OperatorRegistry::add(const std::string &name, OperatorFactory factory) {
  m_factories[name] = std::move(factory);
}

bool OperatorRegistry::has(const std::string &name) const {
  return m_factories.count(name) > 0;
}

std::vector<std::string> OperatorRegistry::names() const {
  std::vector<std::string> out;
  for (const auto &kv : m_factories) {
    out.push_back(kv.first);
  }
  return out;
}

std::unique_ptr<Operator>
OperatorRegistry::create(const config::Section &entry,
                         const ModelInfo &info) const {
  const auto name = entry.string("operator");
  const auto it = m_factories.find(name);
  if (it == m_factories.end()) {
    std::string known;
    for (const auto &n : names()) {
      known += (known.empty() ? "" : ", ") + n;
    }
    throw std::invalid_argument(
        entry.where() + ": no operator '" + name + "' is registered; known: " +
        known + ". A component library registers its own before creating a "
        "model.");
  }
  return it->second(entry, info);
}

// ---------------------------------------------------------------------------
// exchange.publish
// ---------------------------------------------------------------------------

ExchangePublish::ExchangePublish(const config::Section &options,
                                 const ModelInfo &info)
    : m_buffer(info.geometry->num_local()) {
  options.only({"operator", "fields", "clip_min_zero"});
  const auto map = options.section("fields");
  const auto clip = options.names("clip_min_zero");
  for (const auto &name : map.keys()) {
    m_fields.emplace_back(name, FieldRef::parse(map.string(name),
                                                map.where() + "." + name));
    m_clip.push_back(std::find(clip.begin(), clip.end(), name) != clip.end());
  }
  for (const auto &c : clip) {
    if (std::none_of(m_fields.begin(), m_fields.end(),
                     [&](const auto &f) { return f.first == c; })) {
      throw std::invalid_argument(options.where() + ".clip_min_zero: '" + c +
                                  "' is not one of the published fields.");
    }
  }
}

void ExchangePublish::exports(const StepInfo &, Fields &f) {
  if (f.exchange == nullptr) {
    throw std::logic_error("exchange.publish: the model has no exchange.");
  }
  for (std::size_t i = 0; i < m_fields.size(); ++i) {
    const auto from = m_fields[i].second.read(f);
    std::copy(from.begin(), from.end(), m_buffer.begin());
    if (m_clip[i]) {
      for (auto &v : m_buffer) {
        v = std::max(v, 0.0);
      }
    }
    f.exchange->publish(m_fields[i].first, m_buffer);
  }
}

// ---------------------------------------------------------------------------
// Window means
// ---------------------------------------------------------------------------

WindowMeanForcing::WindowMeanForcing(const config::Section &options,
                                     const ModelInfo &info,
                                     std::vector<std::string> channels)
    : m_channels(std::move(channels)),
      m_clip_min_zero(options.names("clip_min_zero")),
      m_suffix(options.string_or("also_into_suffix", "")),
      m_window(m_channels, info.geometry->num_local()),
      m_sample(info.geometry->num_local()) {
  if (info.layout == nullptr) {
    throw std::invalid_argument(options.where() +
                                ": a window mean needs a network to force.");
  }
  for (const auto &c : m_channels) {
    m_sample.add(c);
  }
  for (const auto &c : m_clip_min_zero) {
    if (std::find(m_channels.begin(), m_channels.end(), c) ==
        m_channels.end()) {
      throw std::invalid_argument(options.where() + ".clip_min_zero: '" + c +
                                  "' is not one of the channels.");
    }
  }
}

Declarations WindowMeanForcing::declarations() const {
  Declarations d;
  for (const auto &c : m_channels) {
    d.writes_inputs.push_back(c);
    if (!m_suffix.empty()) {
      d.writes_inputs.push_back(c + m_suffix);
    }
  }
  return d;
}

void WindowMeanForcing::sample(const StepInfo &, Fields &f) {
  fill_sample(f, m_sample);
  m_window.add(m_sample);
}

void WindowMeanForcing::before_step(const StepInfo &, Fields &f) {
  auto &in = *f.inputs;
  for (const auto &name : m_channels) {
    m_window.mean(name, in.get(name));
  }
  for (const auto &name : m_channels) {
    if (std::find(m_clip_min_zero.begin(), m_clip_min_zero.end(), name) !=
        m_clip_min_zero.end()) {
      for (auto &v : in.get(name)) {
        v = std::max(v, 0.0);
      }
    }
    if (!m_suffix.empty()) {
      const auto from = in.get(name);
      auto next = in.get(name + m_suffix);
      std::copy(from.begin(), from.end(), next.begin());
    }
  }
}

void WindowMeanForcing::after_step(const StepInfo &, Fields &) {
  m_window.reset();
}

void WindowMeanForcing::save_to(coupling::RestartStore &store,
                                const std::string &prefix) const {
  m_window.save_to(store, prefix);
}

void WindowMeanForcing::load_from(coupling::RestartStore &store,
                                  const std::string &prefix) {
  m_window.load_from(store, prefix);
}

ExchangeWindowMean::ExchangeWindowMean(const config::Section &options,
                                       const ModelInfo &info)
    : WindowMeanForcing(options, info, options.names("channels")),
      m_prefix(options.string("prefix")) {
  options.only({"operator", "prefix", "channels", "also_into_suffix",
                "clip_min_zero"});
}

void ExchangeWindowMean::fill_sample(Fields &f, fields::FieldSet &sample) {
  if (f.exchange == nullptr) {
    throw std::logic_error("exchange.window_mean: the model has no exchange.");
  }
  for (const auto &name : channels()) {
    const auto from = f.exchange->get(m_prefix + name);
    auto to = sample.get(name);
    std::copy(from.begin(), from.end(), to.begin());
  }
}

// ---------------------------------------------------------------------------
// insolation
// ---------------------------------------------------------------------------

namespace {

physics::Orbit read_orbit(const config::Section &options) {
  const auto o = options.section("orbit");
  o.only({"eccen", "obliq", "mvelp"});
  return physics::Orbit::from_elements(o.number("eccen"), o.number("obliq"),
                                       o.number("mvelp"));
}

} // namespace

InsolationOperator::InsolationOperator(const config::Section &options,
                                       const ModelInfo &info)
    : m_channel(options.string("channel")),
      m_sun(read_orbit(options), info.geometry->lat, info.geometry->lon) {
  options.only({"operator", "channel", "orbit"});
  if (info.layout == nullptr) {
    throw std::invalid_argument(options.where() +
                                ": insolation needs a network input to set.");
  }
}

Declarations InsolationOperator::declarations() const {
  Declarations d;
  d.aux = {"solin_window", "solin_now"};
  d.writes_inputs = {m_channel};
  return d;
}

void InsolationOperator::window(const StepInfo &info, Fields &f) {
  auto solin = f.inputs->get(m_channel);
  m_sun.window_mean(info.now.ymd, info.now.tod, info.model_dt, solin);
  auto held = f.aux->get("solin_window");
  std::copy(solin.begin(), solin.end(), held.begin());
}

void InsolationOperator::initialize(const StepInfo &info, Fields &f) {
  window(info, f);
}

void InsolationOperator::before_step(const StepInfo &info, Fields &f) {
  window(info, f);
}

void InsolationOperator::exports(const StepInfo &info, Fields &f) {
  m_sun.instantaneous(info.now.ymd, info.now.tod, f.aux->get("solin_now"));
}

void register_common_operators(OperatorRegistry &registry) {
  registry.add("exchange.publish", [](const config::Section &o,
                                      const ModelInfo &i) {
    return std::make_unique<ExchangePublish>(o, i);
  });
  registry.add("exchange.window_mean", [](const config::Section &o,
                                          const ModelInfo &i) {
    return std::make_unique<ExchangeWindowMean>(o, i);
  });
  registry.add("insolation", [](const config::Section &o, const ModelInfo &i) {
    return std::make_unique<InsolationOperator>(o, i);
  });
}

} // namespace model
} // namespace emulator
