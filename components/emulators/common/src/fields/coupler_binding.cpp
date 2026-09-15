/**
 * @file coupler_binding.cpp
 * @brief Implementation of AttrVectView and CouplerBinding.
 */

#include "coupler_binding.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace emulator {
namespace fields {

namespace {

std::string join(const std::vector<std::string> &names) {
  std::string out;
  for (const auto &n : names) {
    out += (out.empty() ? "" : " ") + n;
  }
  return out.empty() ? "(none)" : out;
}

const char *direction_name(CouplerBinding::Direction d) {
  return d == CouplerBinding::Direction::Import ? "import" : "export";
}

} // namespace

// ===========================================================================
// AttrVectView
// ===========================================================================

AttrVectView::AttrVectView(double *data, const FieldList &layout,
                           std::size_t nfields, std::size_t npoints)
    : m_data(data), m_layout(&layout), m_nfields(nfields),
      m_npoints(npoints) {
  if (layout.size() != nfields) {
    std::ostringstream oss;
    oss << "The coupler's attribute vector has " << nfields
        << " real fields but its field list names " << layout.size() << ".";
    if (layout.size() < nfields) {
      oss << " A short list is usually a truncated one: check that the "
             "seq_flds_*_fields string was not copied into a fixed-length "
             "buffer on the way here. It ends with '"
          << (layout.empty() ? std::string() : layout.names().back()) << "'.";
    }
    throw std::invalid_argument(oss.str());
  }
  if (data == nullptr && npoints > 0 && nfields > 0) {
    throw std::invalid_argument(
        "The coupler's attribute vector has " + std::to_string(npoints) +
        " points but its data pointer is null.");
  }
}

void AttrVectView::read(std::size_t field, std::span<double> dst) const {
  for (std::size_t p = 0; p < m_npoints; ++p) {
    dst[p] = m_data[p * m_nfields + field];
  }
}

void AttrVectView::write(std::size_t field, std::span<const double> src) {
  for (std::size_t p = 0; p < m_npoints; ++p) {
    m_data[p * m_nfields + field] = src[p];
  }
}

void AttrVectView::fill(std::size_t field, double value) {
  for (std::size_t p = 0; p < m_npoints; ++p) {
    m_data[p * m_nfields + field] = value;
  }
}

// ===========================================================================
// CouplerBinding
// ===========================================================================

CouplerBinding::CouplerBinding(Direction direction, FieldSet &fields,
                               const std::vector<FieldSpec> &specs,
                               const FieldList &coupler_fields,
                               const MaskSet *masks)
    : m_direction(direction), m_fields(&fields),
      m_coupler_fields(&coupler_fields) {
  std::vector<bool> row_bound(coupler_fields.size(), false);
  std::vector<std::string> missing;

  for (const auto &spec : specs) {
    if (!fields.contains(spec.name)) {
      throw std::logic_error("The " + std::string(direction_name(direction)) +
                             " spec names '" + spec.name +
                             "', which the component never added to its "
                             "field set.");
    }
    const auto row = coupler_fields.find(spec.name);
    if (!row) {
      if (spec.need == Need::Required) {
        std::string entry = spec.name;
        const auto near = coupler_fields.near_misses(spec.name);
        if (!near.empty()) {
          entry += " (did you mean " + join(near) + "?)";
        }
        missing.push_back(entry);
      } else {
        m_absent.push_back(spec.name);
      }
      continue;
    }
    if (row_bound[*row]) {
      throw std::invalid_argument("Field '" + spec.name + "' is named twice in "
                                  "the " + direction_name(direction) +
                                  " specs.");
    }
    std::span<const double> mask;
    if (!spec.mask.empty()) {
      if (direction == Direction::Import) {
        throw std::invalid_argument("Import field '" + spec.name +
                                    "' names a mask; only exports are "
                                    "masked.");
      }
      if (masks == nullptr || !masks->contains(spec.mask)) {
        throw std::invalid_argument(
            "Export field '" + spec.name + "' is bounded by mask '" +
            spec.mask + "', which has not been set. Masks are part of the "
            "component's domain and must exist before coupling is set up.");
      }
      mask = masks->get(spec.mask);
      if (mask.size() != fields.npoints()) {
        throw std::invalid_argument("Mask '" + spec.mask + "' has " +
                                    std::to_string(mask.size()) +
                                    " points; the fields have " +
                                    std::to_string(fields.npoints()) + ".");
      }
      m_masked.push_back(spec.name + " (" + spec.mask + ")");
    }
    row_bound[*row] = true;
    m_pairs.push_back({fields.get(spec.name), *row, mask});
    m_bound.push_back(spec.name);
  }

  if (!missing.empty()) {
    std::ostringstream oss;
    if (direction == Direction::Import) {
      oss << "The component needs coupler fields the coupler does not send: ";
    } else {
      oss << "The component exports fields the coupler does not carry, so "
             "this component is paired with the wrong compset or coupler "
             "field list: ";
    }
    for (std::size_t i = 0; i < missing.size(); ++i) {
      oss << (i ? ", " : "") << missing[i];
    }
    oss << ". The coupler's list is: " << coupler_fields.to_string();
    throw std::invalid_argument(oss.str());
  }

  for (std::size_t row = 0; row < coupler_fields.size(); ++row) {
    if (!row_bound[row]) {
      m_unbound.push_back(coupler_fields.name(row));
      m_zero_rows.push_back(row);
    }
  }
}

void CouplerBinding::check_view(const AttrVectView &coupler) const {
  if (&coupler.layout() != m_coupler_fields &&
      coupler.layout().names() != m_coupler_fields->names()) {
    throw std::invalid_argument(
        std::string("The attribute vector handed to this ") +
        direction_name(m_direction) +
        " binding is laid out differently from the one it was bound to.");
  }
  if (coupler.npoints() != m_fields->npoints()) {
    throw std::invalid_argument(
        std::string("The coupler's ") + direction_name(m_direction) +
        " attribute vector has " + std::to_string(coupler.npoints()) +
        " points but the component's fields have " +
        std::to_string(m_fields->npoints()) + ".");
  }
}

void CouplerBinding::pull(const AttrVectView &coupler) {
  if (m_direction != Direction::Import) {
    throw std::logic_error("pull() on an export binding.");
  }
  check_view(coupler);
  for (const auto &pair : m_pairs) {
    coupler.read(pair.row, pair.field);
  }
}

void CouplerBinding::push(AttrVectView &coupler) const {
  if (m_direction != Direction::Export) {
    throw std::logic_error("push() on an import binding.");
  }
  check_view(coupler);
  for (const auto &pair : m_pairs) {
    coupler.write(pair.row, pair.field);
    if (!pair.mask.empty()) {
      for (std::size_t p = 0; p < pair.mask.size(); ++p) {
        if (pair.mask[p] == 0.0) {
          coupler.at(pair.row, p) = 0.0;
        }
      }
    }
  }
  for (const auto row : m_zero_rows) {
    coupler.fill(row, 0.0);
  }
}

std::string CouplerBinding::summary() const {
  std::ostringstream oss;
  const bool import = m_direction == Direction::Import;
  oss << direction_name(m_direction) << ": " << m_bound.size() << " of "
      << m_coupler_fields->size() << " coupler fields bound\n"
      << "  bound:   " << join(m_bound) << "\n";
  if (!m_masked.empty()) {
    oss << "  masked:  " << join(m_masked) << "\n";
  }
  if (!m_absent.empty()) {
    oss << "  absent (optional, kept at fill): " << join(m_absent) << "\n";
  }
  oss << (import ? "  ignored: " : "  zeroed:  ") << join(m_unbound) << "\n";
  return oss.str();
}

} // namespace fields
} // namespace emulator
