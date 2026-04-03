#include "conditional_sampling.hpp"
#include "share/util/eamxx_universal_constants.hpp"
#include <ekat_team_policy_utils.hpp>
#include <string>

namespace scream {

// Utility function to apply conditional sampling logic
KOKKOS_INLINE_FUNCTION
bool evaluate_condition(const Real &condition_val, const int &op_code, const Real &comparison_val) {
  // op_code: 0=eq, 1=ne, 2=gt, 3=ge, 4=lt, 5=le
  switch (op_code) {
    case 0: return condition_val == comparison_val;  // eq or ==
    case 1: return condition_val != comparison_val;  // ne or !=
    case 2: return condition_val > comparison_val;   // gt or >
    case 3: return condition_val >= comparison_val;  // ge or >=
    case 4: return condition_val < comparison_val;   // lt or <
    case 5: return condition_val <= comparison_val;  // le or <=
    default: return false;
  }
}

// Utility function to convert operator string to code
int get_operator_code(const std::string& op) {
  if (op == "eq" || op == "==") return 0;
  if (op == "ne" || op == "!=") return 1;
  if (op == "gt" || op == ">")  return 2;
  if (op == "ge" || op == ">=") return 3;
  if (op == "lt" || op == "<")  return 4;
  if (op == "le" || op == "<=") return 5;
  return -1; // Invalid operator
}

// Unified conditional sampling function that works for any rank (1D or 2D)
// by operating on flat data pointers.  This replaces the former 4 separate
// functions (apply_conditional_sampling_{1d,2d,1d_lev,2d_lev}) that were
// nearly identical aside from view rank and condition source.
//
// Parameters:
//   use_lev_condition - if true, condition value is the level index (not a field)
//   nlevs            - number of vertical levels (used for lev condition; 1 for 1D lev)
void apply_conditional_sampling(
    const Field &output_field, const Field &input_field,
    const Field *condition_field_ptr,  // nullptr when use_lev_condition==true
    const std::string &condition_op, const Real &condition_val,
    const bool use_lev_condition, const int nlevs_for_lev_cond,
    const Real &fill_value = constants::fill_value<Real>) {

  const auto is_counting = (fill_value == 0);

  // Get flat data pointers (rank-agnostic)
  auto* output_data = output_field.get_internal_view_data<Real>();
  auto* mask_data = !is_counting
      ? output_field.get_header().get_extra_data<Field>("mask_data").get_internal_view_data<Real>()
      : output_data;
  const auto* input_data = input_field.get_internal_view_data<const Real>();

  // Input mask (flat pointer)
  bool has_input_mask = input_field.get_header().has_extra_data("mask_data");
  const auto* input_mask_data = has_input_mask
      ? input_field.get_header().get_extra_data<Field>("mask_data").get_internal_view_data<const Real>()
      : input_data;

  // Condition field (flat pointer); only used when !use_lev_condition
  const Real* cond_data = nullptr;
  const Real* cond_mask_data = nullptr;
  bool has_condition_mask = false;
  if (!use_lev_condition && condition_field_ptr != nullptr) {
    cond_data = condition_field_ptr->get_internal_view_data<const Real>();
    has_condition_mask = condition_field_ptr->get_header().has_extra_data("mask_data");
    cond_mask_data = has_condition_mask
        ? condition_field_ptr->get_header().get_extra_data<Field>("mask_data").get_internal_view_data<const Real>()
        : cond_data;
  }

  const int total = output_field.get_header().get_identifier().get_layout().size();
  const int op_code = get_operator_code(condition_op);
  const int nlevs = nlevs_for_lev_cond;

  Kokkos::parallel_for(
      "ConditionalSampling", Kokkos::RangePolicy<>(0, total),
      KOKKOS_LAMBDA(const int &i) {
        bool input_masked = has_input_mask && (input_mask_data[i] == 0);

        // Determine condition value for this element
        Real cond_v;
        bool condition_masked = false;
        if (use_lev_condition) {
          // Level-based: for 1D fields nlevs==1 so lev_idx==i,
          // for 2D fields lev_idx == i % nlevs
          cond_v = static_cast<Real>(nlevs > 1 ? (i % nlevs) : i);
        } else {
          cond_v = cond_data[i];
          condition_masked = has_condition_mask && (cond_mask_data[i] == 0);
        }

        if (input_masked || condition_masked) {
          output_data[i] = fill_value;
          if (!is_counting) mask_data[i] = 0;
        } else if (evaluate_condition(cond_v, op_code, condition_val)) {
          output_data[i] = input_data[i];
          if (!is_counting) mask_data[i] = 1;
        } else {
          output_data[i] = fill_value;
          if (!is_counting) mask_data[i] = 0;
        }
      });
}

ConditionalSampling::ConditionalSampling(const ekat::Comm &comm, const ekat::ParameterList &params)
    : AtmosphereDiagnostic(comm, params) {

  m_input_f      = m_params.get<std::string>("input_field");
  m_condition_f  = m_params.get<std::string>("condition_field");
  m_condition_op = m_params.get<std::string>("condition_operator");

  const auto str_condition_v = m_params.get<std::string>("condition_value");
  // TODO: relying on std::stod to throw if bad val is given
  m_condition_v = static_cast<Real>(std::stod(str_condition_v));

  m_diag_name =
      m_input_f + "_where_" + m_condition_f + "_" + m_condition_op + "_" + str_condition_v;
}

void ConditionalSampling::create_requests() {
  const auto &gn = m_params.get<std::string>("grid_name");
  const auto g   = m_grids_manager->get_grid("physics");

  // Special case: if the input field is "count", we don't need to add it
  if (m_input_f != "count") {
    add_field<Required>(m_input_f, gn);
  }

  // Special case: if condition field is "lev", we don't add it as a required field
  // since it's geometric information from the grid, not an actual field
  if (m_condition_f != "lev") {
    add_field<Required>(m_condition_f, gn);
  } else {
    // Store grid info for potential use in count operations
    m_nlevs = g->get_num_vertical_levels();
    m_gn = gn;
  }
}

void ConditionalSampling::initialize_impl(const RunType /*run_type*/) {

  if (m_input_f != "count") {
    auto ifid = get_field_in(m_input_f).get_header().get_identifier();
    m_diagnostic_output = Field(ifid.clone(m_diag_name));
    m_diagnostic_output.allocate_view();
  } else {
    if (m_condition_f != "lev") {
      auto ifid = get_field_in(m_condition_f).get_header().get_identifier();
      m_diagnostic_output = Field(ifid.clone(m_diag_name));
      m_diagnostic_output.allocate_view();
    } else {
      using namespace ShortFieldTagsNames;
      using namespace ekat::units;
      const auto nondim = Units::nondimensional();
      FieldIdentifier d_fid(m_diag_name, {{LEV},{m_nlevs}}, nondim, m_gn);
      m_diagnostic_output = Field(d_fid);
      m_diagnostic_output.allocate_view();
    }
  }

  auto ifid = m_diagnostic_output.get_header().get_identifier();
  Field diag_mask(ifid.clone(m_diag_name + "_mask"));
  diag_mask.allocate_view();

  const auto var_fill_value = constants::fill_value<Real>;
  m_mask_val = m_params.get<double>("mask_value", var_fill_value);
  if (m_input_f != "count") {
    m_diagnostic_output.get_header().set_extra_data("mask_data", diag_mask);
    m_diagnostic_output.get_header().set_extra_data("mask_value", m_mask_val);
  }
  // Special case: if the input field is "count", let's create a field of 1s
  if (m_input_f == "count") {
    ones = m_diagnostic_output.clone("count_ones");
    ones.deep_copy(1.0);
  }
  
  // Special case: if condition field is "lev", we don't need to check layout compatibility
  // since "lev" is geometric information, not an actual field
  if (m_condition_f == "lev") {
    using namespace ShortFieldTagsNames;
    EKAT_REQUIRE_MSG(ifid.get_layout().tags().back() == LEV,
                   "Error! ConditionalSampling with \"lev\" condition field must have level in layout.\n"
                   " - field name: " + ifid.name() + "\n"
                   " - field layout: " + ifid.get_layout().to_string() + "\n");
  } else {
    const auto cfid = get_field_in(m_condition_f).get_header().get_identifier();
    
    // check that m_input_f and m_condition_f have the same layout
    EKAT_REQUIRE_MSG(ifid.get_layout() == cfid.get_layout(),
                     "Error! ConditionalSampling only supports comparing fields of the same layout.\n"
                     " - input field has layout of " + ifid.get_layout().to_string() + "\n" +
                     " - condition field has layout of " + cfid.get_layout().to_string() + "\n");
  }
}

void ConditionalSampling::compute_diagnostic_impl() {
  Field f;
  if (m_input_f == "count") {
    // Special case: if the input field is "count", we use the diagnostic output as the input
    f = ones;
  } else {
    f = get_field_in(m_input_f);
  }
  const auto &d = m_diagnostic_output;

  // Validate operator
  const int op_code = get_operator_code(m_condition_op);
  EKAT_REQUIRE_MSG(op_code >= 0,
                   "Error! Invalid condition operator: '" + m_condition_op + "'\n"
                   "Valid operators are: eq, ==, ne, !=, gt, >, ge, >=, lt, <, le, <=\n");

  // Get the fill value from constants
  const Real fill_value = (m_input_f == "count") ? 0.0 : m_mask_val;

  // Determine field layout and apply appropriate conditional sampling
  const auto &layout = f.get_header().get_identifier().get_layout();
  const int rank     = layout.rank();

  if (rank > 2) {
      // no support for now, contact devs
      EKAT_ERROR_MSG("Error! ConditionalSampling only supports 1D or 2D field layouts.\n"
                     " - field layout: " + layout.to_string() + "\n"
                     " - field rank: " + std::to_string(rank) + "\n");
  }
  if (m_condition_f == "lev") {
    // Level-based: nlevs_for_lev_cond controls how level indices are computed
    // For 1D fields (rank==1), nlevs=1 so level_idx==flat_idx.
    // For 2D fields (rank==2), level_idx = flat_idx % nlevs.
    const int nlevs_lev = (rank == 2) ? layout.dims()[1] : 1;
    apply_conditional_sampling(d, f, nullptr, m_condition_op, m_condition_v,
                               true, nlevs_lev, fill_value);
  } else {
    const auto &c = get_field_in(m_condition_f);
    apply_conditional_sampling(d, f, &c, m_condition_op, m_condition_v,
                               false, 0, fill_value);
  }
}

} // namespace scream
