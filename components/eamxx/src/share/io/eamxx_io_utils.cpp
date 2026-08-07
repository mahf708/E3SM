#include "share/io/eamxx_io_utils.hpp"

#include "share/io/eamxx_diag_names.hpp"
#include "share/scorpio_interface/eamxx_scorpio_interface.hpp"
#include "share/util/eamxx_utils.hpp"
#include "share/core/eamxx_config.hpp"

#include <ekat_string_utils.hpp>

#include <fstream>
#include <regex>

namespace scream {

std::string find_filename_in_rpointer (
    const std::string& filename_prefix,
    const bool model_restart,
    const ekat::Comm& comm,
    const util::TimeStamp& run_t0,
    const bool allow_not_found,
    const OutputAvgType avg_type,
    const IOControl& control)
{
  std::string filename;
  bool found = false;
  std::string content;
  std::string suffix = model_restart ? ".r." : ".rhist.";
  std::string pattern_str = filename_prefix + suffix;

  // The AD will pass a default constructed control, since it doesn't know the values
  // of REST_N/REST_OPTION used in the previous run. Also, model restart is *always* INSTANT.
  if (model_restart) {
    EKAT_REQUIRE_MSG (avg_type==OutputAvgType::Instant,
        "Error! Model restart output should have INSTANT avg type.\n"
        " - input avg_type: " + e2str(avg_type) + "\n");
    pattern_str += e2str(OutputAvgType::Instant) + R"(.n(step|sec|min|hour|day|month|year)s_x\d+)";
  } else {
    EKAT_REQUIRE_MSG (control.output_enabled(),
        "Error! When restarting an output stream, we need a valid IOControl structure.\n"
        " - filename prefix: " + filename_prefix + "\n");
    pattern_str += e2str(avg_type) + "." + control.frequency_units + "_x" + std::to_string(control.frequency);
  }
  if (is_scream_standalone()) {
    pattern_str += ".np" + std::to_string(comm.size());
  }
  pattern_str += "." + run_t0.to_string() + ".nc";
  std::regex pattern (pattern_str);

  if (comm.am_i_root()) {
    std::ifstream rpointer_file;

    std::string line;
    rpointer_file.open("rpointer.atm");

    while (std::getline(rpointer_file,line)) {
      content += line + "\n";

      if (std::regex_search(line,pattern)) {
        filename = line;
        found = true;
        break;
      }
    }
  }

  int ifound = int(found);
  comm.broadcast(&ifound,1,0);
  found = bool(ifound);

  if (found) {
    // Have the root rank communicate the nc filename
    broadcast_string(filename,comm,comm.root_rank());
  } else if (not allow_not_found) {
    broadcast_string(content,comm,comm.root_rank());

    if (model_restart) {
      EKAT_ERROR_MSG (
          "Error! Restart requested, but no model restart file found in 'rpointer.atm'.\n"
          "   model restart filename prefix: " + filename_prefix + "\n"
          "   model restart filename pattern: " + pattern_str + "\n"
          "   run t0           : " + run_t0.to_string() + "\n"
          "   rpointer content:\n" + content + "\n\n");
    } else {
      EKAT_ERROR_MSG (
          "Error! Restart requested, but no history restart file found in 'rpointer.atm'.\n"
          "   hist restart filename prefix: " + filename_prefix + "\n"
          "   hist restart filename pattern: " + pattern_str + "\n"
          "   run t0           : " + run_t0.to_string() + "\n"
          "   avg_type         : " + e2str(avg_type) + "\n"
          "   output freq      : " + std::to_string(control.frequency) + "\n"
          "   output freq units: " + control.frequency_units + "\n"
          "   rpointer content:\n" + content + "\n\n"
          " Did you change output specs (avg type, freq, or freq units) across restart? If so, please, remember that it is not allowed.\n"
          " It is also possible you are using a rhist file create before commit 6b7d441330d. That commit changed how rhist file names\n"
          " are formed. In particular, we no longer use INSTANT.${REST_OPTION}_x${REST_N}, but we use the avg type, and freq/freq_option\n"
          " of the output stream (to avoid name clashes if 2 streams only differ for one of those). If you want to use your rhist file,\n"
          " please rename it, so that the avg-type, freq, and freq_option reflect those of the output stream.\n");
    }
  }

  return filename;
}

std::shared_ptr<AbstractDiagnostic>
create_diagnostic (const std::string& diag_field_name,
                   const std::shared_ptr<const AbstractGrid>& grid)
{
  auto& factory = DiagnosticFactory::instance();

  // Parse the request as a DSL expression and resolve it to a diagnostic name
  // plus params. Sub-expressions are NOT built here: they are named by their
  // canonical form and left for the IO layer to request in turn, which is how
  // diagnostics already compose.
  //
  // The resolution proper lives in eamxx_diag_names.cpp, which knows nothing
  // about EAMxx and can therefore be tested without a model build. The factory
  // lookup is the one thing it cannot do, so it takes it as a callback.
  diag_dsl::DiagSpec spec;
  try {
    spec = diag_dsl::resolve(diag_field_name,grid->name(),
                             [&](const std::string& n) { return factory.has_product(n); });
  } catch (const std::exception& e) {
    EKAT_ERROR_MSG (
        "Error! Could not create the requested diagnostic.\n"
        "   requested: " + diag_field_name + "\n"
        + e.what() + "\n");
  }

  EKAT_REQUIRE_MSG (spec.diag_name!="AeroComCld",
      "Error! AeroComCld diags are disabled for now. Contact developers.\n"
      "      Some recent development made the code produce bad values,\n"
      "      even runtime aborts due to NaNs.\n"
      "      An alternative is to request variables like cdnc_at_cldtop,\n"
      "      which remain unaffected and scientifically valid.\n");

  ekat::ParameterList params(diag_field_name);
  for (const auto& [key,value] : spec.params) {
    params.set<std::string>(key,value);
  }

  // Pin the output field name to what was asked for. The IO layer resolves a
  // diag's dependencies by looking their names up in the field manager, so a
  // diag must publish under the name its requester used -- whether that is a
  // DSL expression or an old composite name. Keeping the requested name is
  // also what keeps existing runs writing identical netCDF variable names.
  params.set<std::string>("output_name",diag_field_name);

  return factory.create(spec.diag_name,grid->get_comm(),params,grid);
}

} // namespace scream
