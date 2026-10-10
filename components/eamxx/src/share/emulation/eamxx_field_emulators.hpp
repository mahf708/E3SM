#ifndef EAMXX_FIELD_EMULATORS_HPP
#define EAMXX_FIELD_EMULATORS_HPP

#include "share/emulation/eamxx_process_emulator.hpp"
#include "share/field/field.hpp"

#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace scream
{

/*
 * Emulators of a whole atmosphere process (or group of processes), working on
 * its fields.
 *
 * Parameters (in the parameter list of the process):
 *   field_emulators: [emu1, ...]   run in this order, each with its own sublist
 *                                  (see eamxx_process_emulator.hpp)
 *   field_emulators_mode: replace | after
 *     replace: the emulators run instead of the process
 *     after:   the emulators run after the process, e.g., to correct it
 *              (mode: add), or to replace some of its outputs
 *
 * Inputs are the fields the process requires or computes, by name (for an
 * output, as the process, or the emulators before, left it), and, in after
 * mode, X_before for an output X: its value before the process ran. Targets
 * are the fields the process computes or updates. Fields go to the emulators
 * in place, with their padding: per column (COL) fields as (ncol), column
 * fields (COL, LEV|ILEV) as (ncol, nlev), and vector fields (COL, CMP, LEV) as
 * one array per component, named <field>_<component index>. Fields of groups
 * are included. Fields that are not Real, or have no COL dimension, are not.
 * A field name present on several grids is also available as <name>@<grid>.
 */
class FieldEmulators
{
public:
  FieldEmulators (const std::string& process_name, const ekat::ParameterList& params);

  // Whether the process params ask for field emulators
  static bool requested (const ekat::ParameterList& params);

  bool replaces_process () const { return m_replace; }

  // Collect the fields of the process (call once, after the fields are allocated)
  void set_fields (const std::list<Field>& fields_in, const std::list<Field>& fields_out);

  // Before the process runs: save the X_before inputs
  void pre_run ();

  // Run the emulators (after the process, or instead of it)
  void run ();

  const std::vector<std::shared_ptr<ProcessEmulator>>& emulators () const { return m_emulators; }

  // Add the arrays of a field to a map, under name (see above for the naming)
  static void add_arrays (ProcessEmulator::arrays_t& arrays, const Field& f, const std::string& name);

private:
  std::string m_process_name;
  bool m_replace = true;
  std::vector<std::shared_ptr<ProcessEmulator>> m_emulators;
  ProcessEmulator::arrays_t m_inputs, m_targets;
  // Output fields, and copies of their values before the process ran (if needed)
  std::vector<std::pair<Field, Field>> m_before;
};

} // namespace scream

#endif // EAMXX_FIELD_EMULATORS_HPP
