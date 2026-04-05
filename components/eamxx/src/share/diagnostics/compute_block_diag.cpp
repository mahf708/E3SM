#include "compute_block_diag.hpp"

#include "share/physics/physics_constants.hpp"
#include "share/field/field_utils.hpp"

namespace scream {

// Create a constant resolver that maps physics constant names to their values
static diag_utils::ConstResolver make_physics_resolver() {
  return [](const std::string& name, double& val) -> bool {
    const auto& dict = physics::Constants<Real>::dictionary();
    ekat::CaseInsensitiveString ci_name(name);
    if (dict.count(ci_name) > 0) {
      val = dict.at(ci_name).value;
      return true;
    }
    return false;
  };
}

ComputeBlockDiag::
ComputeBlockDiag(const ekat::Comm &comm, const ekat::ParameterList &params)
 : AtmosphereDiagnostic(comm, params)
{
  std::string diag_name = m_params.get<std::string>("diag_name");
  std::string compute_str = m_params.get<std::string>("compute");
  std::vector<std::string> inputs;
  if (m_params.isType<std::vector<std::string>>("inputs")) {
    inputs = m_params.get<std::vector<std::string>>("inputs");
  }

  // Auto-detect inputs if not provided: gather all unique field names required
  // across all statements that are NOT already targets of previous statements.
  if (inputs.empty()) {
    std::set<std::string> known;
    auto resolver = make_physics_resolver();
    // Re-parse statements minimally to find unknown variables
    std::istringstream stream(compute_str);
    std::string line;
    while (std::getline(stream, line)) {
      auto start = line.find_first_not_of(" \t");
      if (start == std::string::npos || line[start] == '#') continue;
      auto eq_pos = line.find('=');
      if (eq_pos != std::string::npos && (eq_pos + 1 >= line.size() || line[eq_pos + 1] != '=')) {
        if (eq_pos > 0 && (line[eq_pos-1] == '!' || line[eq_pos-1] == '<' || line[eq_pos-1] == '>')) continue;
        std::string target = line.substr(0, eq_pos);
        std::string rhs = line.substr(eq_pos + 1);
        auto strip = [](const std::string& s) {
          auto a = s.find_first_not_of(" \t");
          if (a == std::string::npos) return std::string("");
          auto b = s.find_last_not_of(" \t");
          return s.substr(a, b - a + 1);
        };
        target = strip(target);
        rhs = strip(rhs);

        std::string expr = rhs;
        if (rhs.substr(0, 8) == "col_sum(" && rhs.back() == ')') expr = rhs.substr(8, rhs.size() - 9);
        else if (rhs.substr(0, 8) == "col_avg(" && rhs.back() == ')') expr = rhs.substr(8, rhs.size() - 9);
        else if (rhs.substr(0, 10) == "horiz_avg(" && rhs.back() == ')') expr = rhs.substr(10, rhs.size() - 11);

        auto parsed = diag_utils::parse_expression(expr, resolver);
        for (const auto& fname : parsed.field_names) {
          if (known.count(fname) == 0) {
            inputs.push_back(fname);
            known.insert(fname);
          }
        }
        known.insert(target);
      }
    }
  }

  // Now properly parse the compute block
  m_block = diag_utils::parse_compute_block(diag_name, inputs, compute_str, make_physics_resolver());
}

void ComputeBlockDiag::
create_requests()
{
  const auto &gname = m_params.get<std::string>("grid_name");
  for (const auto& fname : m_block.input_names) {
    add_field<Required>(fname, gname);
  }
}

void ComputeBlockDiag::initialize_impl(const RunType /*run_type*/)
{
  const auto &gname = m_params.get<std::string>("grid_name");

  // Map inputs
  for (const auto& fname : m_block.input_names) {
    m_vars[fname] = get_field_in(fname);
  }

  // Determine base layout from the first input
  EKAT_REQUIRE_MSG(!m_block.input_names.empty(), "Error! ComputeBlockDiag requires at least one input field.");
  const auto& base_layout = m_vars[m_block.input_names.front()].get_header().get_identifier().get_layout();

  m_evaluators.resize(m_block.statements.size());

  for (size_t i = 0; i < m_block.statements.size(); ++i) {
    const auto& stmt = m_block.statements[i];

    // Create the evaluator for the right-hand side expression
    auto evaluator = std::make_unique<diag_utils::EamxxExpressionEvaluator>(stmt.expression, make_physics_resolver());
    m_evaluators[i] = std::move(evaluator);

    // Determine target layout
    FieldLayout target_layout = base_layout;
    if (stmt.type == diag_utils::StmtType::ColSum || stmt.type == diag_utils::StmtType::ColAvg) {
      EKAT_REQUIRE_MSG(base_layout.has_tag(ShortFieldTagsNames::LEV), "Error! Vertical reduction requires LEV dimension.");
      target_layout = target_layout.clone().strip_dim(ShortFieldTagsNames::LEV);
    } else if (stmt.type == diag_utils::StmtType::HorizAvg) {
      EKAT_REQUIRE_MSG(base_layout.has_tag(ShortFieldTagsNames::COL), "Error! Horizontal reduction requires COL dimension.");
      target_layout = target_layout.clone().strip_dim(ShortFieldTagsNames::COL);
    }

    // Determine units (for now, nondimensional unless explicitly set for result)
    auto units = ekat::units::Units::nondimensional();

    // Check if target is "result" and if there are specific units from params
    if (stmt.target_var == "result") {
      // Allow params to override units
    }

    // Check if we need to create the field
    if (m_vars.find(stmt.target_var) == m_vars.end()) {
      FieldIdentifier fid(stmt.target_var, target_layout, units, gname);
      Field f(fid, true);
      if (stmt.target_var == "result") {
        m_diagnostic_output = f;
      }
      m_vars[stmt.target_var] = f;
    }
  }
}

void ComputeBlockDiag::compute_diagnostic_impl()
{
  for (size_t i = 0; i < m_block.statements.size(); ++i) {
    const auto& stmt = m_block.statements[i];
    auto& evaluator = m_evaluators[i];

    // Gather input fields for the expression from m_vars
    std::vector<const Field*> input_fields;
    for (const auto& fname : evaluator->field_names()) {
      EKAT_REQUIRE_MSG(m_vars.count(fname) > 0, "Error! ComputeBlockDiag: variable '" + fname + "' not found.");
      input_fields.push_back(&m_vars[fname]);
    }

    if (stmt.type == diag_utils::StmtType::Assign) {
      // Evaluate directly into the target field
      evaluator->evaluate(input_fields, m_vars[stmt.target_var]);
    } else {
      // It's a reduction. First evaluate the expression into a temporary field with the base layout.
      const auto& base_layout = input_fields.empty() ? m_vars[stmt.target_var].get_header().get_identifier().get_layout() // This won't work well without inputs
                                                     : input_fields.front()->get_header().get_identifier().get_layout();
      FieldIdentifier tmp_fid("tmp_expr", base_layout, ekat::units::Units::nondimensional(), m_vars[stmt.target_var].get_header().get_identifier().get_grid_name());
      Field tmp_expr(tmp_fid, true);
      evaluator->evaluate(input_fields, tmp_expr);

      if (stmt.type == diag_utils::StmtType::ColSum || stmt.type == diag_utils::StmtType::ColAvg) {
        // Create an unweighted ones field
        FieldLayout wts_layout = { {ShortFieldTagsNames::LEV}, {base_layout.dim(ShortFieldTagsNames::LEV)} };
        FieldIdentifier wts_fid("wts_ones", wts_layout, ekat::units::Units::nondimensional(), tmp_expr.get_header().get_identifier().get_grid_name());
        Field wts(wts_fid, true);
        wts.deep_copy(scream::Real(1.0));

        bool avg = (stmt.type == diag_utils::StmtType::ColAvg);
        scream::vert_contraction(m_vars[stmt.target_var], tmp_expr, wts, avg);
      } else if (stmt.type == diag_utils::StmtType::HorizAvg) {
        // Use horiz_avg (assuming unweighted for now, as area_weights might not be available)
        // TODO: horiz_avg requires specific handling or a dedicated utility
        // For simplicity right now we can do a naive column average
        // wait, we have horiz_avg.hpp/cpp in share/diagnostics but it's a diagnostic itself.
        // It's better to reuse its logic or call a utility.
        // Actually field_utils.hpp doesn't have a horiz_avg yet, but let's see.
        EKAT_ERROR_MSG("ComputeBlockDiag: horiz_avg not yet fully integrated.");
      }
    }
  }
}

}  // namespace scream
