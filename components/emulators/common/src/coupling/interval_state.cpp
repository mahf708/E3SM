/**
 * @file interval_state.cpp
 * @brief Implementation of IntervalMean and BracketedState.
 */

#include "interval_state.hpp"

#include <algorithm>
#include <stdexcept>

namespace emulator {
namespace coupling {

namespace {

std::size_t find_name(const std::vector<std::string> &names,
                      std::string_view name, const char *what) {
  const auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end()) {
    std::string held;
    for (const auto &n : names) {
      held += (held.empty() ? "" : ", ") + n;
    }
    throw std::out_of_range(std::string(what) + " has no channel '" +
                            std::string(name) + "'; it has: " + held + ".");
  }
  return static_cast<std::size_t>(it - names.begin());
}

void check_points(const fields::FieldSet &fields, std::size_t npoints,
                  const char *what) {
  if (fields.npoints() != npoints) {
    throw std::invalid_argument(std::string(what) + " holds " +
                                std::to_string(npoints) +
                                " points per channel; the fields have " +
                                std::to_string(fields.npoints()) + ".");
  }
}

void check_unique(const std::vector<std::string> &names, const char *what) {
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty() ||
        std::find(names.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                  names.end(), names[i]) != names.end()) {
      throw std::invalid_argument(std::string(what) +
                                  ": channel names must be non-empty and "
                                  "unique; '" +
                                  names[i] + "' is not.");
    }
  }
}

} // namespace

// ===========================================================================
// IntervalMean
// ===========================================================================

IntervalMean::IntervalMean(std::vector<std::string> names, std::size_t npoints)
    : m_names(std::move(names)), m_npoints(npoints),
      m_sums(m_names.size(), std::vector<double>(npoints, 0.0)) {
  check_unique(m_names, "An interval mean");
}

void IntervalMean::add(const fields::FieldSet &fields) {
  check_points(fields, m_npoints, "An interval mean");
  // Look every channel up before adding any, so a missing one leaves the
  // sums untouched rather than half a step ahead.
  std::vector<std::span<const double>> sources;
  sources.reserve(m_names.size());
  for (const auto &name : m_names) {
    sources.push_back(fields.get(name));
  }
  for (std::size_t c = 0; c < m_names.size(); ++c) {
    auto &sum = m_sums[c];
    for (std::size_t p = 0; p < m_npoints; ++p) {
      sum[p] += sources[c][p];
    }
  }
  ++m_samples;
}

std::size_t IntervalMean::index_of(std::string_view name) const {
  return find_name(m_names, name, "The interval mean");
}

void IntervalMean::mean(std::string_view name, std::span<double> out) const {
  const auto c = index_of(name);
  if (m_samples == 0) {
    throw std::logic_error("The interval mean of '" + std::string(name) +
                           "' was asked for with no samples in the interval.");
  }
  if (out.size() != m_npoints) {
    throw std::invalid_argument("The interval mean of '" + std::string(name) +
                                "' has " + std::to_string(m_npoints) +
                                " points; the destination has " +
                                std::to_string(out.size()) + ".");
  }
  const double inv = 1.0 / static_cast<double>(m_samples);
  for (std::size_t p = 0; p < m_npoints; ++p) {
    out[p] = m_sums[c][p] * inv;
  }
}

void IntervalMean::reset() {
  for (auto &sum : m_sums) {
    std::fill(sum.begin(), sum.end(), 0.0);
  }
  m_samples = 0;
}

void IntervalMean::save_to(RestartStore &store,
                           std::string_view prefix) const {
  store.write_int(restart_name(prefix, "samples"), m_samples);
  for (std::size_t c = 0; c < m_names.size(); ++c) {
    store.write_array(restart_name(prefix, "sum." + m_names[c]), m_sums[c]);
  }
}

bool IntervalMean::load_from(RestartStore &store, std::string_view prefix,
                             Missing missing) {
  std::int64_t samples = 0;
  const bool have_count = store.read_int(restart_name(prefix, "samples"),
                                         samples);
  if (!have_count) {
    if (missing == Missing::StartEmpty) {
      reset();
      return false;
    }
    throw std::runtime_error("The restart has no '" +
                             restart_name(prefix, "samples") +
                             "'; the interval mean cannot be restored.");
  }
  // With the count present every sum must be too: a partial accumulator is
  // not a consistent state to start from, whatever `missing` says.
  std::vector<std::vector<double>> sums = m_sums;
  for (std::size_t c = 0; c < m_names.size(); ++c) {
    const auto name = restart_name(prefix, "sum." + m_names[c]);
    if (!store.read_array(name, sums[c])) {
      throw std::runtime_error("The restart has '" +
                               restart_name(prefix, "samples") +
                               "' but no '" + name + "'.");
    }
  }
  m_sums = std::move(sums);
  m_samples = static_cast<int>(samples);
  return true;
}

// ===========================================================================
// BracketedState
// ===========================================================================

BracketedState::BracketedState(std::vector<std::string> names,
                               std::size_t npoints)
    : BracketedState(names,
                     std::vector<Temporal>(names.size(), Temporal::Snapshot),
                     npoints) {}

BracketedState::BracketedState(std::vector<std::string> names,
                               std::vector<Temporal> kinds,
                               std::size_t npoints)
    : m_names(std::move(names)), m_kinds(std::move(kinds)), m_npoints(npoints),
      m_lower(m_names.size(), std::vector<double>(npoints, 0.0)),
      m_upper(m_names.size(), std::vector<double>(npoints, 0.0)) {
  check_unique(m_names, "A bracketed state");
  if (m_kinds.size() != m_names.size()) {
    throw std::invalid_argument(
        "A bracketed state of " + std::to_string(m_names.size()) +
        " channels given " + std::to_string(m_kinds.size()) + " kinds.");
  }
}

std::size_t BracketedState::index_of(std::string_view name) const {
  return find_name(m_names, name, "The bracketed state");
}

void BracketedState::copy_in(const fields::FieldSet &from,
                             std::vector<std::vector<double>> &to,
                             const char *what) const {
  check_points(from, m_npoints, what);
  std::vector<std::span<const double>> sources;
  for (const auto &name : m_names) {
    sources.push_back(from.get(name));
  }
  for (std::size_t c = 0; c < m_names.size(); ++c) {
    std::copy(sources[c].begin(), sources[c].end(), to[c].begin());
  }
}

void BracketedState::set_both(const fields::FieldSet &state) {
  copy_in(state, m_lower, "A bracketed state");
  m_upper = m_lower;
  m_seeded = true;
}

void BracketedState::advance(const fields::FieldSet &new_upper) {
  if (!m_seeded) {
    throw std::logic_error("advance() on a bracketed state that was never "
                           "seeded with set_both().");
  }
  std::vector<std::vector<double>> next = m_lower;
  copy_in(new_upper, next, "A bracketed state");
  m_lower.swap(m_upper);
  m_upper.swap(next);
}

void BracketedState::blend(double f, fields::FieldSet &out) const {
  if (!m_seeded) {
    throw std::logic_error("blend() on a bracketed state that was never "
                           "seeded with set_both().");
  }
  if (!(f >= 0.0 && f <= 1.0)) {
    throw std::invalid_argument("A blend weight of " + std::to_string(f) +
                                " is outside [0, 1].");
  }
  check_points(out, m_npoints, "A bracketed state");
  std::vector<std::span<double>> dests;
  for (const auto &name : m_names) {
    dests.push_back(out.get(name));
  }
  for (std::size_t c = 0; c < m_names.size(); ++c) {
    const auto &lo = m_lower[c];
    const auto &hi = m_upper[c];
    auto dst = dests[c];
    if (!m_interpolate || m_kinds[c] == Temporal::IntervalMean) {
      std::copy(hi.begin(), hi.end(), dst.begin());
      continue;
    }
    for (std::size_t p = 0; p < m_npoints; ++p) {
      dst[p] = lo[p] + f * (hi[p] - lo[p]);
    }
  }
}

std::span<const double> BracketedState::lower(std::string_view name) const {
  return m_lower[index_of(name)];
}

std::span<const double> BracketedState::upper(std::string_view name) const {
  return m_upper[index_of(name)];
}

void BracketedState::save_to(RestartStore &store,
                             std::string_view prefix) const {
  store.write_int(restart_name(prefix, "seeded"), m_seeded ? 1 : 0);
  for (std::size_t c = 0; c < m_names.size(); ++c) {
    store.write_array(restart_name(prefix, "lower." + m_names[c]), m_lower[c]);
    store.write_array(restart_name(prefix, "upper." + m_names[c]), m_upper[c]);
  }
}

void BracketedState::load_from(RestartStore &store, std::string_view prefix) {
  std::int64_t seeded = 0;
  if (!store.read_int(restart_name(prefix, "seeded"), seeded)) {
    throw std::runtime_error("The restart has no '" +
                             restart_name(prefix, "seeded") + "'.");
  }
  auto lower = m_lower;
  auto upper = m_upper;
  for (std::size_t c = 0; c < m_names.size(); ++c) {
    for (auto [which, dest] : {std::pair{"lower.", &lower[c]},
                               std::pair{"upper.", &upper[c]}}) {
      const auto name = restart_name(prefix, which + m_names[c]);
      if (!store.read_array(name, *dest)) {
        throw std::runtime_error(
            "The restart has no '" + name +
            "'. Both brackets are needed: restarting the interpolation from "
            "one state changes what the coupler sees for the rest of the "
            "interval.");
      }
    }
  }
  m_lower = std::move(lower);
  m_upper = std::move(upper);
  m_seeded = seeded != 0;
}

} // namespace coupling
} // namespace emulator
