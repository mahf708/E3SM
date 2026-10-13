#ifndef EAMXX_VERT_CONTRACT_HPP
#define EAMXX_VERT_CONTRACT_HPP

#include "share/diagnostics/abstract_diagnostic.hpp"

namespace scream {

/*
 * This diagnostic will calculate a reduction of a field across the LEV tag
 * dimension, producing an N-1 dimensional field.
 *
 * For "sum":   out = sum_lev(weight * f * mask)
 * For "avg":   out = sum_lev(weight * f * mask) / sum_lev(weight * mask)
 * For "max":   out = max_lev(f) over entries with mask!=0
 * For "min":   out = min_lev(f) over entries with mask!=0
 *
 * The weight is dp/g, dz, or 1 ("none"). Only "sum" and "avg" are weighted:
 * "max" and "min" require weighting_method=none.
 *
 * If the input field has a valid_mask, entries with mask==0 are excluded from
 * the reduction (for "avg", from both numerator and denominator). For "avg",
 * "max" and "min", output entries with no valid level are filled with
 * fill_value and the output field's valid_mask is set to 0.
 */

class VertContract : public AbstractDiagnostic {
 public:
  // Constructors
  VertContract(const ekat::Comm &comm, const ekat::ParameterList &params,
               const std::shared_ptr<const AbstractGrid> &grid);

  // The name of the diagnostic CLASS (not the computed field)
  std::string name() const { return "VertContract"; }

 protected:
#ifdef KOKKOS_ENABLE_CUDA
 public:
#endif
  void compute_impl();
  void initialize_impl();

  std::string m_field_name; // Input field name

  // Name of contraction method (avg, sum, max, min)
  std::string m_contract_method;
  // Name of weighting method (dp, dz, none)
  std::string m_weighting_method;

  // Weight field (dp/g, dz, or all-ones depending on weighting_method)
  Field m_weight;

  // For avg mode: denominator scratch field (same layout as m_diagnostic_output).
  // For max/min of a masked field: weighted count of valid levels, used to
  // detect output entries with no valid level.
  Field m_weight_sum;

  // Ones field with the same layout as the input field (to compute m_weight_sum)
  Field m_ones;
};

}  // namespace scream

#endif  // EAMXX_VERT_CONTRACT_HPP
