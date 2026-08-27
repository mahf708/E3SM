#include "share/diagnostics/abstract_diagnostic.hpp"

#include <ekat_std_utils.hpp>

namespace scream
{

AbstractDiagnostic::
AbstractDiagnostic (const ekat::Comm& comm,
                    const ekat::ParameterList& params,
                    const std::shared_ptr<const AbstractGrid>& grid)
 : m_comm(comm)
 , m_params(params)
 , m_grid(grid)
{
  EKAT_REQUIRE_MSG (grid, "[AbstractDiagnostic] Error! Invalid grid pointer.\n");
}

void AbstractDiagnostic::initialize ()
{
  initialize_impl();

  // Diags name their own output field by concatenating their params, e.g.
  // BinaryOp builds "<arg1>_<op>_<arg2>". That name is not always the one the
  // customer asked for: 'X_atm_backtend' is an alias for 'X_minus_X_prev_over_dt',
  // and a diag built from an expression is named after the expression. Since
  // customers look the field up by the name they requested, honor an explicit
  // request when there is one.
  // NOTE: aliasing shares the header's tracking, alloc props and extra data, so
  //       masks, timestamps and avg-cnt bookkeeping all carry over.
  if (m_params.isParameter("output_field_name")) {
    const auto& name = m_params.get<std::string>("output_field_name");
    if (name!=m_diagnostic_output.name()) {
      m_diagnostic_output = m_diagnostic_output.alias(name);
    }
  }

  m_is_initialized = true;
}

void AbstractDiagnostic::set_input_field (const Field& f)
{
  // Safety check
  EKAT_REQUIRE_MSG(ekat::contains(m_field_in_names,f.name()),
      "Error! Setting a field in the diagnostic that was not requested.\n"
      " - diag name: " + name() + "\n"
      " - field name: " + f.name() + "\n");
  m_fields_in[f.name()] = f;
}

Field AbstractDiagnostic::get () const
{
  EKAT_REQUIRE_MSG (m_diagnostic_output.is_allocated(),
      "Error! Getting a diagnostic field before it is allocated is suspicious at best.\n"
      "       We chose to throw an error, but if this is a legit use, please, contact developers.\n"
      " - diag field name: " + m_diagnostic_output.name() + "\n");
  return m_diagnostic_output;
}

void AbstractDiagnostic::compute (const util::TimeStamp& ts)
{
  // Compute a hash of ts with all the timestamps of the input fields
  bfbhash::HashType tsh = 0;
  for (auto it : m_fields_in) {
    const auto& fts = it.second.get_header().get_tracking().get_time_stamp();
    util::hash(fts,tsh);
  }
  util::hash(ts,tsh);

  // If the hash matches the last evaluation hash, then nothing has really
  // changed, so the stored diagnostic field does not have to be recomputed
  if (tsh==m_last_eval_ts_hash) {
    return;
  }

  compute_impl ();

  // Update timestamp info
  m_diagnostic_output.get_header().get_tracking().update_time_stamp(ts);
  m_last_eval_ts = ts;
  m_last_eval_ts_hash = tsh;
}

} // namespace scream
