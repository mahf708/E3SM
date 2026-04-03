#include "share/io/eamxx_io_utils.hpp"

#include "share/scorpio_interface/eamxx_scorpio_interface.hpp"
#include "share/data_managers/library_grids_manager.hpp"
#include "share/util/eamxx_utils.hpp"
#include "share/core/eamxx_config.hpp"

#include <ekat_string_utils.hpp>

#include <fstream>
#include <functional>
#include <regex>
#include <vector>

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

void write_timestamp (const std::string& filename, const std::string& ts_name,
                      const util::TimeStamp& ts, const bool write_nsteps)
{
  scorpio::set_attribute(filename,"GLOBAL",ts_name,ts.to_string());
  if (write_nsteps) {
    scorpio::set_attribute(filename,"GLOBAL",ts_name+"_nsteps",ts.get_num_steps());
  }
}

util::TimeStamp read_timestamp (const std::string& filename,
                                const std::string& ts_name,
                                const bool read_nsteps)
{
  auto ts = util::str_to_time_stamp(scorpio::get_attribute<std::string>(filename,"GLOBAL",ts_name));
  if (read_nsteps and scorpio::has_attribute(filename,"GLOBAL",ts_name+"_nsteps")) {
    ts.set_num_steps(scorpio::get_attribute<int>(filename,"GLOBAL",ts_name+"_nsteps"));
  }
  return ts;
}

// ---- Table-driven diagnostic name parser ----
//
// Each DiagSpec defines a regex pattern and a function that extracts
// parameters from the match groups.  This replaces the former monolithic
// if-else chain with a self-documenting table that is easy to extend:
// just append a new entry -- no need to touch any other code.

namespace {

// Generic field name pattern: letters, digits, dash, dot, plus, minus, product, division
const std::string generic_field = "([A-Za-z0-9_.+\\-\\*\\÷]+)";

struct DiagSpec {
  std::string  pattern_str;  // regex source (compiled lazily)
  std::string  diag_name;    // factory key in AtmosphereDiagnosticFactory
  // Given the match groups and the grid, populate `params`.
  // Return false to signal that the diagnostic should NOT be created
  // (e.g. disabled diagnostics that emit an error message).
  std::function<bool(const std::smatch&,
                     const std::shared_ptr<const AbstractGrid>&,
                     ekat::ParameterList&)> extract;
};

// The table is ordered: earlier entries take priority (same semantics as
// the original if-else chain).  Each lambda receives the regex match
// groups and fills the parameter list for the diagnostic constructor.
const std::vector<DiagSpec>& get_diag_specs ()
{
  static const std::vector<DiagSpec> specs = {
    // --- FieldAtLevel: e.g. T_mid_at_lev_5, T_mid_at_model_top ---
    { generic_field + R"(_at_(lev_(\d+)|model_(top|bot))$)",
      "FieldAtLevel",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("field_name",        m[1].str());
        p.set("grid_name",         grid->name());
        p.set("vertical_location", m[2].str());
        return true;
      }
    },
    // --- FieldAtPressureLevel: e.g. T_mid_at_500hPa ---
    { generic_field + R"(_at_(\d+(\.\d+)?)(hPa|mb|Pa)$)",
      "FieldAtPressureLevel",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("field_name",      m[1].str());
        p.set("grid_name",       grid->name());
        p.set("pressure_value",  m[2].str());
        p.set("pressure_units",  m[4].str());
        return true;
      }
    },
    // --- FieldAtHeight: e.g. T_mid_at_100m_above_sealevel ---
    { generic_field + R"(_at_(\d+(\.\d+)?)(m)_above_(sealevel|surface)$)",
      "FieldAtHeight",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("field_name",         m[1].str());
        p.set("grid_name",          grid->name());
        p.set("height_value",       m[2].str());
        p.set("height_units",       m[4].str());
        p.set("surface_reference",  m[5].str());
        return true;
      }
    },
    // --- PrecipSurfMassFlux: e.g. precip_liq_surf_mass_flux ---
    { R"(precip_(liq|ice|total)_surf_mass_flux$)",
      "precip_surf_mass_flux",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>&,
         ekat::ParameterList& p) {
        p.set<std::string>("precip_type", m[1].str());
        return true;
      }
    },
    // --- WaterPath: e.g. LiqWaterPath ---
    { R"((Ice|Liq|Rain|Rime|Vap)WaterPath$)",
      "WaterPath",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>&,
         ekat::ParameterList& p) {
        p.set<std::string>("water_kind", m[1].str());
        return true;
      }
    },
    // --- NumberPath: e.g. IceNumberPath ---
    { R"((Ice|Liq|Rain)NumberPath$)",
      "NumberPath",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>&,
         ekat::ParameterList& p) {
        p.set<std::string>("number_kind", m[1].str());
        return true;
      }
    },
    // --- AeroComCld (disabled): e.g. AeroComCldTop ---
    { R"(AeroComCld(Top|Bot)$)",
      "AeroComCld",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>&,
         ekat::ParameterList& p) {
        EKAT_ERROR_MSG("Error! AeroComCld diags are disabled for now. Contact developers.\n"
                       "      Some recent development made the code produce bad values,\n"
                       "      even runtime aborts due to NaNs.\n"
                       "      An alternative is to request variables like cdnc_at_cldtop,\n"
                       "      which remain unaffected and scientifically valid.\n");
        p.set<std::string>("aero_com_cld_kind", m[1].str());
        return false;
      }
    },
    // --- VaporFlux: e.g. MeridionalVapFlux ---
    { R"((Meridional|Zonal)VapFlux$)",
      "VaporFlux",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>&,
         ekat::ParameterList& p) {
        p.set<std::string>("wind_component", m[1].str());
        return true;
      }
    },
    // --- AtmBackTendDiag: e.g. T_mid_atm_backtend ---
    { generic_field + R"(_atm_backtend$)",
      "AtmBackTendDiag",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("grid_name",              grid->name());
        p.set<std::string>("tendency_name", m[1].str());
        return true;
      }
    },
    // --- PotentialTemperature: e.g. PotentialTemperature, LiqPotentialTemperature ---
    { R"((Liq)?PotentialTemperature$)",
      "PotentialTemperature",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>&,
         ekat::ParameterList& p) {
        p.set<std::string>("temperature_kind",
                           m[1].str() != "" ? m[1].str() : std::string("Tot"));
        return true;
      }
    },
    // --- VerticalLayer: e.g. z_mid, geopotential_int ---
    { R"((z|geopotential|height)_(mid|int)$)",
      "VerticalLayer",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>&,
         ekat::ParameterList& p) {
        p.set<std::string>("diag_name",      m[1].str());
        p.set<std::string>("vert_location",  m[2].str());
        return true;
      }
    },
    // --- HorizAvgDiag: e.g. T_mid_horiz_avg ---
    { generic_field + R"(_horiz_avg$)",
      "HorizAvgDiag",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("grid_name",               grid->name());
        p.set<std::string>("field_name",  m[1].str());
        return true;
      }
    },
    // --- VertContractDiag: e.g. T_mid_vert_avg, T_mid_vert_sum_dp_weighted ---
    { generic_field + R"(_vert_(avg|sum)(_((dp|dz)_weighted))?$)",
      "VertContractDiag",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("grid_name",                     grid->name());
        p.set<std::string>("field_name",        m[1].str());
        p.set<std::string>("contract_method",   m[2].str());
        if (m[3].matched) {
          p.set<std::string>("weighting_method", m[5].str());
        }
        return true;
      }
    },
    // --- VertDerivativeDiag: e.g. T_mid_pvert_derivative ---
    { generic_field + R"(_(p|z)vert_derivative$)",
      "VertDerivativeDiag",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("grid_name",                       grid->name());
        p.set<std::string>("field_name",          m[1].str());
        p.set<std::string>("derivative_method",   m[2].str());
        return true;
      }
    },
    // --- ZonalAvgDiag: e.g. T_mid_zonal_avg_36_bins ---
    { generic_field + R"(_zonal_avg_(\d+)_bins$)",
      "ZonalAvgDiag",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("grid_name",                            grid->name());
        p.set<std::string>("field_name",               m[1].str());
        p.set<std::string>("number_of_zonal_bins",     m[2].str());
        return true;
      }
    },
    // --- ConditionalSampling: e.g. T_mid_where_cldfrac_liq_gt_0.5 ---
    { generic_field + R"(_where_)" + generic_field +
      R"(_(gt|ge|eq|ne|le|lt)_([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$)",
      "ConditionalSampling",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("grid_name",                          grid->name());
        p.set<std::string>("input_field",            m[1].str());
        p.set<std::string>("condition_field",        m[2].str());
        p.set<std::string>("condition_operator",     m[3].str());
        p.set<std::string>("condition_value",        m[4].str());
        return true;
      }
    },
    // --- UnaryOpsDiag: e.g. sqrt_of_T_mid, abs_of_qc ---
    { R"((sqrt|abs|log|exp|square)_of_)" + generic_field + "$",
      "UnaryOpsDiag",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("grid_name",                grid->name());
        p.set<std::string>("unary_op",    m[1].str());
        p.set<std::string>("arg",         m[2].str());
        return true;
      }
    },
    // --- BinaryOpsDiag: e.g. qc_plus_qv, T_mid_times_gravit ---
    { generic_field + R"(_(plus|minus|times|over)_)" + generic_field + "$",
      "BinaryOpsDiag",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("grid_name",               grid->name());
        p.set<std::string>("arg1",        m[1].str());
        p.set<std::string>("arg2",        m[3].str());
        p.set<std::string>("binary_op",   m[2].str());
        return true;
      }
    },
    // --- ExpressionDiag: e.g. expr_qc*pseudo_density/gravit ---
    // Any field name starting with "expr_" is treated as an expression diagnostic.
    // The expression itself is everything after the "expr_" prefix.
    { R"(^expr_(.+)$)",
      "ExpressionDiag",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("grid_name",                  grid->name());
        p.set<std::string>("expression",    m[1].str());
        return true;
      }
    },
    // --- HistogramDiag: e.g. T_mid_histogram_200_250_300 ---
    { generic_field + R"(_histogram_(\d+(\.\d+)?(_\d+(\.\d+)?)+)$)",
      "HistogramDiag",
      [](const std::smatch& m, const std::shared_ptr<const AbstractGrid>& grid,
         ekat::ParameterList& p) {
        p.set("grid_name",                       grid->name());
        p.set<std::string>("field_name",          m[1].str());
        p.set<std::string>("bin_configuration",   m[2].str());
        return true;
      }
    },
  };
  return specs;
}

} // anonymous namespace

std::shared_ptr<AtmosphereDiagnostic>
create_diagnostic (const std::string& diag_field_name,
                   const std::shared_ptr<const AbstractGrid>& grid)
{
  std::string diag_name;
  ekat::ParameterList params(diag_field_name);

  // Try each registered diagnostic pattern in order
  bool matched = false;
  for (const auto& spec : get_diag_specs()) {
    std::smatch matches;
    std::regex pattern(spec.pattern_str);
    if (std::regex_search(diag_field_name, matches, pattern)) {
      diag_name = spec.diag_name;
      spec.extract(matches, grid, params);
      matched = true;
      break;
    }
  }

  // Special case: "dz" is a VerticalLayer diagnostic
  if (!matched && diag_field_name == "dz") {
    diag_name = "VerticalLayer";
    params.set<std::string>("diag_name", "dz");
    params.set<std::string>("vert_location", "mid");
    matched = true;
  }

  if (!matched) {
    // No pattern matched -- assume the field name IS the diagnostic name.
    diag_name = diag_field_name;
  }

  auto comm = grid->get_comm();
  auto diag = AtmosphereDiagnosticFactory::instance().create(diag_name,comm,params);
  auto gm = std::make_shared<LibraryGridsManager>(grid);
  diag->set_grids(gm);

  return diag;
}

} // namespace scream
