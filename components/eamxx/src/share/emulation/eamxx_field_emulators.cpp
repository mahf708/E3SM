#include "share/emulation/eamxx_field_emulators.hpp"

#include "share/util/eamxx_data_type.hpp"

#include <ekat_assert.hpp>

#include <set>

namespace scream
{

bool FieldEmulators::requested (const ekat::ParameterList& params)
{
  return params.isParameter("field_emulators") and
         not params.get<std::vector<std::string>>("field_emulators").empty();
}

FieldEmulators::
FieldEmulators (const std::string& process_name, const ekat::ParameterList& params_in)
 : m_process_name (process_name)
{
  auto params = params_in;
  const auto mode = params.get<std::string>("field_emulators_mode", "replace");
  EKAT_REQUIRE_MSG (mode=="replace" or mode=="after",
      "[FieldEmulators] Error! In process '" + process_name + "', field_emulators_mode must be "
      "'replace' or 'after'.\n");
  m_replace = mode=="replace";

  for (const auto& n : params.get<std::vector<std::string>>("field_emulators")) {
    EKAT_REQUIRE_MSG (params.isSublist(n),
        "[FieldEmulators] Error! Missing parameter sublist for emulator '" + n + "' of process '" +
        process_name + "'.\n");
    m_emulators.push_back(std::make_shared<ProcessEmulator>(n, params.sublist(n)));
  }
}

void FieldEmulators::
add_arrays (ProcessEmulator::arrays_t& arrays, const Field& f, const std::string& name)
{
  if (f.data_type()!=get_data_type<Real>()) {
    return;
  }
  const auto& fl = f.get_header().get_identifier().get_layout();
  if (fl.rank()<1 or fl.tags()[0]!=FieldTag::Column) {
    return;
  }
  // Extents from the layout: the views of fields include the padding
  switch (fl.rank()) {
    case 1: {
      const auto v = f.get_strided_view<const Real*>();
      arrays[name] = ProcessEmulator::strided(v.data(), fl.dim(0), 1, v.stride(0), 1, true);
      break;
    }
    case 2: {
      const auto v = f.get_strided_view<const Real**>();
      arrays[name] = ProcessEmulator::strided(v.data(), fl.dim(0), fl.dim(1), v.stride(0), v.stride(1), false);
      break;
    }
    case 3: {
      const auto v = f.get_strided_view<const Real***>();
      for (int c=0; c<fl.dim(1); ++c) {
        arrays[name + "_" + std::to_string(c)] =
          ProcessEmulator::strided(v.data() + c*v.stride(1), fl.dim(0), fl.dim(2), v.stride(0), v.stride(2), false);
      }
      break;
    }
    default:
      break;
  }
}

void FieldEmulators::set_fields (const std::list<Field>& fields_in, const std::list<Field>& fields_out)
{
  // Names present on several grids are also available as name@grid
  std::map<std::string, std::set<std::string>> grids;
  for (const auto& fl : {&fields_in, &fields_out}) {
    for (const auto& f : *fl) {
      grids[f.name()].insert(f.get_header().get_identifier().get_grid_name());
    }
  }
  // Add the arrays of field data, named after field f
  auto add = [&](ProcessEmulator::arrays_t& arrays, const Field& f, const std::string& suffix,
                 const Field& data) {
    const auto& grid = f.get_header().get_identifier().get_grid_name();
    if (grids[f.name()].size()>1) {
      add_arrays(arrays, data, f.name() + suffix + "@" + grid);
    }
    if (arrays.count(f.name() + suffix)==0) {
      add_arrays(arrays, data, f.name() + suffix);
    }
  };

  m_inputs.clear();
  m_targets.clear();
  for (const auto& f : fields_in) {
    add(m_inputs, f, "", f);
  }
  for (const auto& f : fields_out) {
    add(m_inputs, f, "", f);
    add(m_targets, f, "", f);
  }

  // In after mode, the values of the outputs before the process ran
  std::set<std::string> wanted;
  for (const auto& emu : m_emulators) {
    for (const auto& n : emu->input_names()) wanted.insert(n);
  }
  m_before.clear();
  if (not m_replace) {
    for (const auto& f : fields_out) {
      ProcessEmulator::arrays_t probe;
      add(probe, f, "_before", f);
      bool used = false;
      for (const auto& [n, a] : probe) used |= wanted.count(n)>0;
      if (used) {
        auto copy = f.clone(f.name() + "_before");
        add(m_inputs, f, "_before", copy);
        m_before.emplace_back(f, copy);
      }
    }
  }
}

void FieldEmulators::pre_run ()
{
  for (auto& [f, copy] : m_before) {
    copy.deep_copy(f);
  }
}

void FieldEmulators::run ()
{
  for (const auto& emu : m_emulators) {
    emu->run(m_inputs, m_targets);
  }
}

} // namespace scream
