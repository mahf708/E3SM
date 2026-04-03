#ifndef KOKKOS_DIAG_PYDIAG_HPP
#define KOKKOS_DIAG_PYDIAG_HPP

// ====================================================================
// Python bindings for the standalone expression parser
//
// This provides nanobind bindings so that scientists can:
//   1. Parse and validate expressions from Python
//   2. Inspect the RPN instruction stream
//   3. Evaluate expressions on NumPy arrays (host-side)
//
// Usage from Python:
//   import pyeamxx_ext
//   result = pyeamxx_ext.parse_expression("sqrt(u**2 + v**2)")
//   print(result.field_names)   # ['u', 'v']
//   print(len(result.program))  # 8
//
//   # Evaluate on numpy arrays
//   out = pyeamxx_ext.eval_expression("a + b", {"a": arr_a, "b": arr_b})
// ====================================================================

#include "expression_parser.hpp"
#include "rpn_evaluator.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>

#include <cmath>
#include <stdexcept>

namespace nb = nanobind;

namespace diag_utils {

// Host-side expression evaluator for NumPy arrays
inline nb::ndarray<nb::numpy, double, nb::ndim<1>>
eval_expression_numpy(const std::string& expr_str,
                      nb::dict field_arrays,
                      nb::object const_resolver_py = nb::none()) {
  // Build constant resolver from Python callable (if provided)
  ConstResolver resolver = nullptr;
  if (!const_resolver_py.is_none()) {
    auto py_resolver = nb::cast<nb::callable>(const_resolver_py);
    resolver = [py_resolver](const std::string& name, double& val) -> bool {
      try {
        nb::object result = py_resolver(nb::cast(name));
        if (result.is_none()) return false;
        val = nb::cast<double>(result);
        return true;
      } catch (...) {
        return false;
      }
    };
  }

  // Parse
  auto parsed = resolver ? parse_expression(expr_str, resolver)
                         : parse_expression(expr_str);

  // Validate that all fields are provided
  for (const auto& fname : parsed.field_names) {
    if (!field_arrays.contains(fname.c_str())) {
      throw std::runtime_error(
        "eval_expression: missing field '" + fname + "' in field_arrays dict");
    }
  }

  // Get array pointers and size
  std::vector<const double*> field_ptrs(parsed.field_names.size());
  size_t N = 0;
  for (size_t j = 0; j < parsed.field_names.size(); ++j) {
    auto arr = nb::cast<nb::ndarray<nb::numpy, const double>>(
      field_arrays[parsed.field_names[j].c_str()]);
    field_ptrs[j] = arr.data();
    size_t this_size = 1;
    for (size_t d = 0; d < arr.ndim(); ++d) this_size *= arr.shape(d);
    if (j == 0) N = this_size;
    else if (this_size != N) {
      throw std::runtime_error("eval_expression: all arrays must have the same total size");
    }
  }

  // Allocate output
  double* out_data = new double[N];
  const double* const* fptrs = field_ptrs.data();

  // Evaluate (host-side, sequential)
  for (size_t i = 0; i < N; ++i) {
    out_data[i] = evaluate_rpn(parsed.program.data(),
                               static_cast<int>(parsed.program.size()),
                               fptrs, static_cast<int>(i));
  }

  // Return as NumPy array (takes ownership of out_data)
  size_t shape[1] = {N};
  nb::capsule owner(out_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  return nb::ndarray<nb::numpy, double, nb::ndim<1>>(out_data, 1, shape, owner);
}

// Register Python bindings
inline void nb_pydiag(nb::module_& m) {
  // ExprOp enum
  nb::enum_<ExprOp>(m, "ExprOp")
    .value("PushField", ExprOp::PushField)
    .value("PushConst", ExprOp::PushConst)
    .value("Add",       ExprOp::Add)
    .value("Sub",       ExprOp::Sub)
    .value("Mul",       ExprOp::Mul)
    .value("Div",       ExprOp::Div)
    .value("Pow",       ExprOp::Pow)
    .value("Min",       ExprOp::Min)
    .value("Max",       ExprOp::Max)
    .value("Sqrt",      ExprOp::Sqrt)
    .value("Abs",       ExprOp::Abs)
    .value("Log",       ExprOp::Log)
    .value("Exp",       ExprOp::Exp)
    .value("Square",    ExprOp::Square)
    .value("Neg",       ExprOp::Neg);

  // Instruction struct
  nb::class_<Instruction>(m, "Instruction")
    .def_ro("op",          &Instruction::op)
    .def_ro("operand_idx", &Instruction::operand_idx)
    .def_ro("const_val",   &Instruction::const_val);

  // ParseResult struct
  nb::class_<ParseResult>(m, "ParseResult")
    .def_ro("program",     &ParseResult::program)
    .def_ro("field_names", &ParseResult::field_names);

  // Parse function
  m.def("parse_expression",
    [](const std::string& expr) { return parse_expression(expr); },
    nb::arg("expression"),
    "Parse a diagnostic expression into RPN instructions.\n"
    "Returns a ParseResult with .program (list of Instructions) and .field_names.");

  // Parse with constant resolver
  m.def("parse_expression",
    [](const std::string& expr, nb::callable resolver) {
      ConstResolver cpp_resolver = [resolver](const std::string& name, double& val) -> bool {
        try {
          nb::object result = resolver(nb::cast(name));
          if (result.is_none()) return false;
          val = nb::cast<double>(result);
          return true;
        } catch (...) {
          return false;
        }
      };
      return parse_expression(expr, cpp_resolver);
    },
    nb::arg("expression"), nb::arg("const_resolver"),
    "Parse with a constant resolver callback(name) -> float or None.");

  // Evaluate on numpy arrays
  m.def("eval_expression", &eval_expression_numpy,
    nb::arg("expression"), nb::arg("fields"),
    nb::arg("const_resolver") = nb::none(),
    "Evaluate an expression on numpy arrays.\n"
    "  expression: str - the expression to evaluate\n"
    "  fields: dict[str, ndarray] - field name -> numpy array mapping\n"
    "  const_resolver: callable(name)->float or None\n"
    "Returns a 1D numpy array with the result.");
}

} // namespace diag_utils

#endif // KOKKOS_DIAG_PYDIAG_HPP
