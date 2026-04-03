#ifndef KOKKOS_DIAG_DIAGNOSTIC_BASE_HPP
#define KOKKOS_DIAG_DIAGNOSTIC_BASE_HPP

// ====================================================================
// Generic diagnostic base class
//
// This header is part of the kokkos-diag-utils package.  It provides
// a lightweight base for field diagnostics that does NOT depend on
// any specific model infrastructure (EAMxx, MPAS, OMEGA, etc.).
//
// Model-specific adapters implement the FieldAccessor concept and
// plug it into DiagnosticBase<FieldAccessor>.
//
// The FieldAccessor concept requires:
//   - using scalar_type = ...;
//   - using field_type  = ...;
//   - static const scalar_type* get_data(const field_type& f);
//   - static scalar_type* get_data_mut(field_type& f);
//   - static long long    get_size(const field_type& f);
//   - static std::string  get_name(const field_type& f);
// ====================================================================

#include "expression_parser.hpp"
#include "rpn_evaluator.hpp"

#include <Kokkos_Core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace diag_utils {

// Generic expression-based diagnostic that operates on any field type
// satisfying the FieldAccessor concept.
template <typename FieldAccessor>
class ExpressionEvaluator {
 public:
  using scalar_type = typename FieldAccessor::scalar_type;
  using field_type  = typename FieldAccessor::field_type;

  ExpressionEvaluator(const std::string& expression,
                      const ConstResolver& const_resolver = nullptr)
    : m_expr_str(expression)
  {
    auto result = const_resolver
                  ? parse_expression(expression, const_resolver)
                  : parse_expression(expression);
    m_program     = std::move(result.program);
    m_field_names = std::move(result.field_names);
  }

  // The field names that the expression references
  const std::vector<std::string>& field_names() const { return m_field_names; }

  // The RPN program
  const std::vector<Instruction>& program() const { return m_program; }

  // The original expression string
  const std::string& expression() const { return m_expr_str; }

  // Number of referenced fields
  int num_fields() const { return static_cast<int>(m_field_names.size()); }

  // Evaluate the expression, writing results to output_field.
  // input_fields must be in the same order as field_names().
  void evaluate(const std::vector<const field_type*>& input_fields,
                field_type& output_field) const
  {
    const int n_fields = static_cast<int>(input_fields.size());
    const int n_instr  = static_cast<int>(m_program.size());
    const long long total_size = FieldAccessor::get_size(output_field);

    // Gather device pointers
    const scalar_type* field_ptrs[MAX_EXPR_STACK] = {};
    for (int j = 0; j < n_fields; ++j) {
      field_ptrs[j] = FieldAccessor::get_data(*input_fields[j]);
    }

    // Copy program to device
    Kokkos::View<Instruction*, Kokkos::DefaultExecutionSpace::memory_space>
      d_program("expr_prog", n_instr);
    {
      auto h = Kokkos::create_mirror_view(d_program);
      for (int i = 0; i < n_instr; ++i) h(i) = m_program[i];
      Kokkos::deep_copy(d_program, h);
    }

    // Copy field pointers to device via uintptr_t
    Kokkos::View<uintptr_t*, Kokkos::DefaultExecutionSpace::memory_space>
      d_fptrs("fld_ptrs", n_fields);
    {
      auto h = Kokkos::create_mirror_view(d_fptrs);
      for (int j = 0; j < n_fields; ++j)
        h(j) = reinterpret_cast<uintptr_t>(field_ptrs[j]);
      Kokkos::deep_copy(d_fptrs, h);
    }

    auto d_out = FieldAccessor::get_data_mut(output_field);
    auto prog  = d_program;
    auto fptrs = d_fptrs;
    const int ni = n_instr;
    const int nf = n_fields;

    Kokkos::parallel_for("ExprEval",
      Kokkos::RangePolicy<>(0, total_size),
      KOKKOS_LAMBDA(const int i) {
        const scalar_type* local_fptrs[MAX_EXPR_STACK];
        for (int j = 0; j < nf; ++j)
          local_fptrs[j] = reinterpret_cast<const scalar_type*>(fptrs(j));
        d_out[i] = evaluate_rpn(prog.data(), ni, local_fptrs, i);
      });
  }

 private:
  std::string              m_expr_str;
  std::vector<Instruction> m_program;
  std::vector<std::string> m_field_names;
};

} // namespace diag_utils

#endif // KOKKOS_DIAG_DIAGNOSTIC_BASE_HPP
