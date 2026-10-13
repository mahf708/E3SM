#ifndef EAMXX_HORIZ_AVERAGE_HPP
#define EAMXX_HORIZ_AVERAGE_HPP

#include "share/diagnostics/abstract_diagnostic.hpp"

namespace scream {

/*
 * This diagnostic will calculate a reduction of a field across the COL tag
 * dimension, producing an N-1 dimensional field. The reduction is selected by
 * the optional 'contract_method' param:
 *
 * For "avg" (default): out = sum(area*f*mask) / sum(area*mask)
 * For "sum":           out = sum(f*mask)  (unweighted)
 * For "max":           out = max(f) over entries with mask!=0
 * For "min":           out = min(f) over entries with mask!=0
 *
 * where mask is the input field's valid_mask (or 1 if it has none).
 * For "avg", "max" and "min", output entries with no valid column are set to
 * fill_value, and the output field's valid_mask is set to 0 at those locations.
 */

class HorizAvg : public AbstractDiagnostic {
 public:
  // Constructors
  HorizAvg(const ekat::Comm &comm, const ekat::ParameterList &params,
               const std::shared_ptr<const AbstractGrid>& grid);

  // The name of the diagnostic CLASS (not the computed field)
  std::string name() const { return "HorizAvg"; }

 protected:
#ifdef KOKKOS_ENABLE_CUDA
 public:
#endif
  void compute_impl();

 protected:
  void initialize_impl();

  // Name of contraction method (avg, sum, max, min)
  std::string m_contract_method;

  // Weight field: area for "avg", all-ones otherwise
  Field m_area;

  // Utility fields to compute the (scalar) denominator
  Field m_denom;
  Field m_ones;
};

}  // namespace scream

#endif  // EAMXX_HORIZ_AVERAGE_HPP
