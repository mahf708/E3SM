/**
 * @file model_fields.cpp
 * @brief Implementation of FieldRef.
 */

#include "model_fields.hpp"

#include <stdexcept>

namespace emulator {
namespace model {

Geometry Geometry::from_grid(MPI_Comm comm, const grid::HorizontalGrid &grid,
                             const grid::Decomposition &decomp,
                             std::vector<double> domain_mask) {
  Geometry g;
  g.comm = comm;
  g.grid = &grid;
  g.decomp = decomp;
  g.gather = std::make_shared<grid::GlobalGather>(comm, decomp);
  g.lat = decomp.local(grid.lat);
  g.lon = decomp.local(grid.lon);
  g.area = decomp.local(grid.area);
  g.domain_mask = domain_mask.empty()
                      ? std::vector<double>(decomp.num_local(), 1.0)
                      : std::move(domain_mask);
  if (g.domain_mask.size() != decomp.num_local()) {
    throw std::invalid_argument("Geometry: a domain mask of " +
                                std::to_string(g.domain_mask.size()) +
                                " values for " +
                                std::to_string(decomp.num_local()) + " cells.");
  }
  return g;
}

Geometry Geometry::from_domain(MPI_Comm comm, const grid::Domain &domain,
                               const grid::Decomposition &decomp) {
  if (domain.size() != decomp.num_local()) {
    throw std::invalid_argument("Geometry: a domain of " +
                                std::to_string(domain.size()) + " cells for a "
                                "decomposition of " +
                                std::to_string(decomp.num_local()) + ".");
  }
  Geometry g;
  g.comm = comm;
  g.decomp = decomp;
  g.gather = std::make_shared<grid::GlobalGather>(comm, decomp);
  g.lat = domain.lat;
  g.lon = domain.lon;
  g.area = domain.area;
  g.domain_mask = domain.mask;
  return g;
}

namespace {

const fields::FieldSet *set_of(const Fields &f, FieldRef::Set s) {
  switch (s) {
  case FieldRef::Set::Imports: return f.imports;
  case FieldRef::Set::Exports: return f.exports;
  case FieldRef::Set::Inputs: return f.inputs;
  case FieldRef::Set::State: return f.state;
  case FieldRef::Set::Prediction: return f.prediction;
  case FieldRef::Set::Aux: return f.aux;
  case FieldRef::Set::Statics: return f.statics;
  default: return nullptr;
  }
}

} // namespace

FieldRef FieldRef::parse(const std::string &text, const std::string &where) {
  const auto dot = text.find('.');
  if (dot == std::string::npos || dot + 1 == text.size()) {
    throw std::invalid_argument(
        where + ": '" + text + "' is not a field reference; write <set>.<name> "
        "with set one of imports, exports, inputs, state, prediction, upper, "
        "aux, statics, exchange.");
  }
  const auto set = text.substr(0, dot);
  FieldRef r;
  r.m_text = text;
  r.m_name = text.substr(dot + 1);
  if (set == "imports") r.m_set = Set::Imports;
  else if (set == "exports") r.m_set = Set::Exports;
  else if (set == "inputs") r.m_set = Set::Inputs;
  else if (set == "state") r.m_set = Set::State;
  else if (set == "prediction") r.m_set = Set::Prediction;
  else if (set == "upper") r.m_set = Set::Upper;
  else if (set == "aux") r.m_set = Set::Aux;
  else if (set == "statics") r.m_set = Set::Statics;
  else if (set == "exchange") r.m_set = Set::Exchange;
  else {
    throw std::invalid_argument(where + ": '" + text + "': unknown set '" +
                                set + "'.");
  }
  return r;
}

bool FieldRef::present(const Fields &f) const {
  if (m_set == Set::Exchange) {
    return f.exchange != nullptr && f.exchange->has(m_name);
  }
  if (m_set == Set::Upper) {
    return f.upper != nullptr && f.upper->seeded();
  }
  const auto *s = set_of(f, m_set);
  return s != nullptr && s->contains(m_name);
}

std::span<const double> FieldRef::read(const Fields &f) const {
  if (m_set == Set::Exchange) {
    if (f.exchange == nullptr) {
      throw std::out_of_range("'" + m_text + "': this model has no exchange.");
    }
    return f.exchange->get(m_name);
  }
  if (m_set == Set::Upper) {
    if (f.upper == nullptr) {
      throw std::out_of_range("'" + m_text + "': this model has no network.");
    }
    return f.upper->upper(m_name);
  }
  const auto *s = set_of(f, m_set);
  if (s == nullptr || !s->contains(m_name)) {
    throw std::out_of_range("'" + m_text + "' is not there.");
  }
  return s->get(m_name);
}

std::span<double> FieldRef::write(Fields &f) const {
  if (!writable()) {
    throw std::logic_error("'" + m_text + "' is read-only; operators write "
                           "exports, inputs and aux.");
  }
  auto *s = const_cast<fields::FieldSet *>(set_of(f, m_set));
  if (s == nullptr || !s->contains(m_name)) {
    throw std::out_of_range("'" + m_text + "' is not there.");
  }
  return s->get(m_name);
}

} // namespace model
} // namespace emulator
