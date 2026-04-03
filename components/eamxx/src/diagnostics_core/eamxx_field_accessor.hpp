#ifndef KOKKOS_DIAG_EAMXX_FIELD_ACCESSOR_HPP
#define KOKKOS_DIAG_EAMXX_FIELD_ACCESSOR_HPP

// ====================================================================
// EAMxx adapter for kokkos-diag-utils
//
// This header bridges the standalone diagnostic package to EAMxx's
// Field class.  It implements the FieldAccessor concept required by
// DiagnosticBase<FieldAccessor>.
//
// This is the ONLY file with EAMxx dependencies.  Everything else
// in diagnostics_core/ is model-agnostic.
// ====================================================================

#include "diagnostic_base.hpp"
#include "share/field/field.hpp"

namespace diag_utils {

// FieldAccessor implementation for EAMxx's Field class
struct EamxxFieldAccessor {
  using scalar_type = scream::Real;
  using field_type  = scream::Field;

  static const scalar_type* get_data(const field_type& f) {
    return f.get_internal_view_data<const scalar_type>();
  }

  static scalar_type* get_data_mut(field_type& f) {
    return f.get_internal_view_data<scalar_type>();
  }

  static long long get_size(const field_type& f) {
    return f.get_header().get_identifier().get_layout().size();
  }

  static std::string get_name(const field_type& f) {
    return f.get_header().get_identifier().name();
  }
};

// Convenience alias
using EamxxExpressionEvaluator = ExpressionEvaluator<EamxxFieldAccessor>;

} // namespace diag_utils

#endif // KOKKOS_DIAG_EAMXX_FIELD_ACCESSOR_HPP
