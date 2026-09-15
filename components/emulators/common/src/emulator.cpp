/**
 * @file emulator.cpp
 * @brief Implementation of the Emulator base class.
 */

#include "emulator.hpp"


#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace emulator {

Emulator::Emulator(EmulatorType type, int id, const std::string &name)
    : m_type(type), m_id(id), m_name(name), m_initialized(false),
      m_step_count(0) {}

void Emulator::initialize() {
  if (m_initialized) {
    throw std::runtime_error("Emulator already initialized");
  }
  init_impl();
  // The coupler reads the component's exports before its first run, so the
  // initial state has to be in the buffer when initialization returns.
  push_exports();
  m_initialized = true;
}

void Emulator::run(int dt) { run(dt, coupling::ModelTime{}); }

void Emulator::run(int dt, coupling::ModelTime now) {
  if (!m_initialized) {
    throw std::runtime_error("Emulator::run() called before initialize()");
  }
  m_current_time = now;
  pull_imports();
  run_impl(dt);
  push_exports();
  m_current_time = coupling::ModelTime{};
  m_step_count++;
}

void Emulator::set_domain(grid::Domain domain, int nx, int ny,
                          std::size_t num_global) {
  if (is_coupled()) {
    throw std::logic_error(
        "Emulator '" + m_name +
        "': the grid cannot change after setup_coupling(); the coupler's "
        "attribute vectors were sized from the old one.");
  }
  const auto n = domain.size();
  if (domain.global_ids.size() != n || domain.lon.size() != n ||
      domain.area.size() != n || domain.mask.size() != n ||
      domain.frac.size() != n) {
    throw std::invalid_argument("Emulator '" + m_name +
                                "': a domain whose arrays differ in length.");
  }
  if (n > num_global) {
    throw std::invalid_argument(
        "Emulator '" + m_name + "': " + std::to_string(n) +
        " local cells but only " + std::to_string(num_global) + " global.");
  }
  m_domain = std::move(domain);
  m_masks = fields::MaskSet(n);
  m_nx = nx;
  m_ny = ny;
  m_num_global = num_global;
  m_has_domain = true;
}

void Emulator::set_grid_data(const EmulatorGridDesc &grid) {
  if (grid.num_local_cols < 0 || grid.num_global_cols < 0) {
    throw std::invalid_argument("Emulator '" + m_name +
                                "': negative column count in the grid "
                                "descriptor.");
  }
  const auto n = static_cast<std::size_t>(grid.num_local_cols);
  if (n > 0 && (!grid.col_gids || !grid.lat || !grid.lon || !grid.area)) {
    throw std::invalid_argument("Emulator '" + m_name +
                                "': the grid descriptor has columns but a "
                                "null coordinate array.");
  }
  grid::Domain d;
  if (n > 0) {
    d.global_ids.assign(grid.col_gids, grid.col_gids + n);
    d.lat.assign(grid.lat, grid.lat + n);
    d.lon.assign(grid.lon, grid.lon + n);
    d.area.assign(grid.area, grid.area + n);
  }
  d.mask.assign(n, 1.0);
  d.frac.assign(n, 1.0);
  set_domain(std::move(d), grid.nx, grid.ny,
             static_cast<std::size_t>(grid.num_global_cols));
}

int Emulator::get_num_local_cols() const {
  return static_cast<int>(m_domain.size());
}
int Emulator::get_num_global_cols() const {
  return static_cast<int>(m_num_global);
}
int Emulator::get_nx() const { return m_nx; }
int Emulator::get_ny() const { return m_ny; }

void Emulator::get_local_col_gids(int *gids) const {
  std::copy(m_domain.global_ids.begin(), m_domain.global_ids.end(), gids);
}

void Emulator::get_cols_latlon(double *lat, double *lon) const {
  std::copy(m_domain.lat.begin(), m_domain.lat.end(), lat);
  std::copy(m_domain.lon.begin(), m_domain.lon.end(), lon);
}

void Emulator::get_cols_area(double *area) const {
  std::copy(m_domain.area.begin(), m_domain.area.end(), area);
}

void Emulator::get_cols_mask_frac(double *mask, double *frac) const {
  std::copy(m_domain.mask.begin(), m_domain.mask.end(), mask);
  std::copy(m_domain.frac.begin(), m_domain.frac.end(), frac);
}

void Emulator::set_coupler_field_lists(std::string_view import_fields,
                                       std::string_view export_fields) {
  m_import_list = fields::FieldList::parse(import_fields);
  m_export_list = fields::FieldList::parse(export_fields);
  m_have_field_lists = true;
}

void Emulator::setup_coupling(const EmulatorCouplingDesc &cpl) {
  if (!m_have_field_lists) {
    throw std::logic_error(
        "Emulator '" + m_name +
        "': setup_coupling() before set_coupler_field_lists(). The field "
        "names are what the buffers are bound by.");
  }
  if (cpl.field_size < 0 || cpl.num_imports < 0 || cpl.num_exports < 0) {
    throw std::invalid_argument("Emulator '" + m_name +
                                "': negative size in the coupling descriptor.");
  }
  if (cpl.field_size != get_num_local_cols()) {
    throw std::invalid_argument(
        "Emulator '" + m_name + "': the coupler's attribute vectors have " +
        std::to_string(cpl.field_size) + " points, but this rank owns " +
        std::to_string(get_num_local_cols()) +
        " columns. The gsMap and the component disagree about the "
        "decomposition.");
  }

  // Check both lists against their vectors before building anything.
  const auto npoints = static_cast<std::size_t>(cpl.field_size);
  fields::AttrVectView(cpl.import_data, m_import_list,
                       static_cast<std::size_t>(cpl.num_imports), npoints);
  fields::AttrVectView(cpl.export_data, m_export_list,
                       static_cast<std::size_t>(cpl.num_exports), npoints);

  const auto wanted = coupling_fields();
  fields::FieldSet imports(npoints);
  fields::FieldSet exports(npoints);
  for (const auto &spec : wanted.imports) {
    imports.add(spec.name);
  }
  for (const auto &spec : wanted.exports) {
    exports.add(spec.name);
  }

  m_cpl = cpl;
  m_imports = std::move(imports);
  m_exports = std::move(exports);
  m_import_binding.emplace(fields::CouplerBinding::Direction::Import,
                           m_imports, wanted.imports, m_import_list);
  m_export_binding.emplace(fields::CouplerBinding::Direction::Export,
                           m_exports, wanted.exports, m_export_list,
                           &m_masks);
}

void Emulator::pull_imports() {
  if (!m_import_binding) {
    return;
  }
  const fields::AttrVectView view(
      m_cpl.import_data, m_import_list,
      static_cast<std::size_t>(m_cpl.num_imports),
      static_cast<std::size_t>(m_cpl.field_size));
  m_import_binding->pull(view);
}

void Emulator::push_exports() {
  if (!m_export_binding) {
    return;
  }
  fields::AttrVectView view(m_cpl.export_data, m_export_list,
                            static_cast<std::size_t>(m_cpl.num_exports),
                            static_cast<std::size_t>(m_cpl.field_size));
  m_export_binding->push(view);
}

void Emulator::finalize() {
  if (!m_initialized) {
    return; // Already finalized or never initialized
  }
  final_impl();
  m_initialized = false;
}


static std::string to_string(EmulatorType t) {
  switch (t) {
  case EmulatorType::ATM_COMP: return "ATM_COMP";
  case EmulatorType::OCN_COMP: return "OCN_COMP";
  case EmulatorType::ICE_COMP: return "ICE_COMP";
  case EmulatorType::LND_COMP: return "LND_COMP";
  default: return "UNKNOWN";
  }
}

void Emulator::print_info(std::ostream& os) const {
  os << "Emulator '" << m_name << "'\n";
  os << "  type          : " << to_string(m_type) << "\n";
  os << "  id            : " << m_id << "\n";
  os << "  initialized   : " << std::boolalpha << m_initialized << "\n";
  os << "  step_count    : " << m_step_count << "\n";

  int nx    = get_nx();
  int ny    = get_ny();
  int nloc  = get_num_local_cols();
  int nglob = get_num_global_cols();

  os << "  grid          : nx=" << nx
     << " ny=" << ny
     << " num_local_cols=" << nloc
     << " num_global_cols=" << nglob << "\n";

  print_extra_info(os);
}

} // namespace emulator
