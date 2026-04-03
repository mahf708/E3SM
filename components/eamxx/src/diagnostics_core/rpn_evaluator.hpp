#ifndef KOKKOS_DIAG_RPN_EVALUATOR_HPP
#define KOKKOS_DIAG_RPN_EVALUATOR_HPP

// ====================================================================
// GPU-safe RPN (Reverse Polish Notation) expression evaluator
//
// This header is part of the kokkos-diag-utils package.
// Dependencies: Kokkos (for KOKKOS_INLINE_FUNCTION), <cmath>
//
// The evaluator runs a stack machine to evaluate an expression
// program produced by expression_parser.hpp.  It is designed to
// be called from within a Kokkos parallel_for kernel.
//
// Usage in a kernel:
//   Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {
//     out[i] = diag_utils::evaluate_rpn(program, n_instr, field_ptrs, i);
//   });
// ====================================================================

#include "expression_parser.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

namespace diag_utils {

// Evaluate an RPN program at element index `elem_idx`.
//
// Parameters:
//   program    - array of Instruction (on device)
//   n_instr    - number of instructions
//   field_ptrs - array of pointers to field data arrays (on device)
//   elem_idx   - the flat element index to evaluate at
//
// Returns the scalar result for this element.
KOKKOS_INLINE_FUNCTION
double evaluate_rpn(const Instruction* program, const int n_instr,
                    const double* const* field_ptrs, const int elem_idx) {
  double stack[MAX_EXPR_STACK];
  int sp = 0;

  for (int pc = 0; pc < n_instr; ++pc) {
    const auto& instr = program[pc];
    switch (instr.op) {
      case ExprOp::PushField:
        stack[sp++] = field_ptrs[instr.operand_idx][elem_idx];
        break;
      case ExprOp::PushConst:
        stack[sp++] = instr.const_val;
        break;

      // Binary ops
      case ExprOp::Add: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=a+b; break; }
      case ExprOp::Sub: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=a-b; break; }
      case ExprOp::Mul: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=a*b; break; }
      case ExprOp::Div: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=a/b; break; }
      case ExprOp::Pow: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=pow(a,b); break; }
      case ExprOp::Min: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a<b?a:b); break; }
      case ExprOp::Max: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a>b?a:b); break; }

      // Comparison ops (return 1.0 for true, 0.0 for false)
      case ExprOp::CmpGt: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a> b?1.0:0.0); break; }
      case ExprOp::CmpGe: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a>=b?1.0:0.0); break; }
      case ExprOp::CmpLt: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a< b?1.0:0.0); break; }
      case ExprOp::CmpLe: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a<=b?1.0:0.0); break; }
      case ExprOp::CmpEq: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a==b?1.0:0.0); break; }
      case ExprOp::CmpNe: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a!=b?1.0:0.0); break; }

      // Ternary: where(cond, true_val, false_val)
      case ExprOp::Where: { auto f=stack[--sp]; auto t=stack[--sp]; auto c=stack[--sp];
                             stack[sp++]=(c!=0.0?t:f); break; }

      // Unary ops
      case ExprOp::Sqrt:   { auto a=stack[--sp]; stack[sp++]=sqrt(a);  break; }
      case ExprOp::Abs:    { auto a=stack[--sp]; stack[sp++]=fabs(a);  break; }
      case ExprOp::Log:    { auto a=stack[--sp]; stack[sp++]=log(a);   break; }
      case ExprOp::Exp:    { auto a=stack[--sp]; stack[sp++]=exp(a);   break; }
      case ExprOp::Square: { auto a=stack[--sp]; stack[sp++]=a*a;      break; }
      case ExprOp::Neg:    { auto a=stack[--sp]; stack[sp++]=-a;       break; }
      case ExprOp::Log10:  { auto a=stack[--sp]; stack[sp++]=log(a)/log(10.0); break; }
    }
  }
  return stack[0];
}

// Float overload for single-precision codes
KOKKOS_INLINE_FUNCTION
float evaluate_rpn_f(const Instruction* program, const int n_instr,
                     const float* const* field_ptrs, const int elem_idx) {
  float stack[MAX_EXPR_STACK];
  int sp = 0;

  for (int pc = 0; pc < n_instr; ++pc) {
    const auto& instr = program[pc];
    switch (instr.op) {
      case ExprOp::PushField:
        stack[sp++] = field_ptrs[instr.operand_idx][elem_idx];
        break;
      case ExprOp::PushConst:
        stack[sp++] = static_cast<float>(instr.const_val);
        break;
      case ExprOp::Add: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=a+b; break; }
      case ExprOp::Sub: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=a-b; break; }
      case ExprOp::Mul: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=a*b; break; }
      case ExprOp::Div: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=a/b; break; }
      case ExprOp::Pow: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=powf(a,b); break; }
      case ExprOp::Min: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a<b?a:b); break; }
      case ExprOp::Max: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a>b?a:b); break; }
      case ExprOp::CmpGt: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a> b?1.0f:0.0f); break; }
      case ExprOp::CmpGe: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a>=b?1.0f:0.0f); break; }
      case ExprOp::CmpLt: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a< b?1.0f:0.0f); break; }
      case ExprOp::CmpLe: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a<=b?1.0f:0.0f); break; }
      case ExprOp::CmpEq: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a==b?1.0f:0.0f); break; }
      case ExprOp::CmpNe: { auto b=stack[--sp]; auto a=stack[--sp]; stack[sp++]=(a!=b?1.0f:0.0f); break; }
      case ExprOp::Where: { auto f=stack[--sp]; auto t=stack[--sp]; auto c=stack[--sp];
                             stack[sp++]=(c!=0.0f?t:f); break; }
      case ExprOp::Sqrt:   { auto a=stack[--sp]; stack[sp++]=sqrtf(a);  break; }
      case ExprOp::Abs:    { auto a=stack[--sp]; stack[sp++]=fabsf(a);  break; }
      case ExprOp::Log:    { auto a=stack[--sp]; stack[sp++]=logf(a);   break; }
      case ExprOp::Exp:    { auto a=stack[--sp]; stack[sp++]=expf(a);   break; }
      case ExprOp::Square: { auto a=stack[--sp]; stack[sp++]=a*a;       break; }
      case ExprOp::Neg:    { auto a=stack[--sp]; stack[sp++]=-a;        break; }
      case ExprOp::Log10:  { auto a=stack[--sp]; stack[sp++]=logf(a)/logf(10.0f); break; }
    }
  }
  return stack[0];
}

} // namespace diag_utils

#endif // KOKKOS_DIAG_RPN_EVALUATOR_HPP
