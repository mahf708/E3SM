#ifndef SCREAM_FIELD_UTILS_HPP
#define SCREAM_FIELD_UTILS_HPP

#include "share/util/eamxx_scalar_wrapper.hpp"
#include "share/field/field.hpp"

namespace scream {

// Check that two fields store the same entries.
//  - fields MUST have the same layout
//  - if either field (or both) is masked, only entries with nonzero mask are compared
//  - if a field is padded, entries in the padding are NOT checked
//  - the fields are compared pointwise with the provided (absolute) tolerance
bool views_are_equal(const Field& f1, const Field& f2, const Real tol = 0);

// Uniform or normal randomization utils
void randomize_uniform (const Field& f, int seed,
                        const ScalarWrapper lb = ScalarWrapper::zero(),
                        const ScalarWrapper ub = ScalarWrapper::one());
void randomize_normal (const Field& f, int seed,
                       const ScalarWrapper mean = ScalarWrapper::zero(),
                       const ScalarWrapper sigma = ScalarWrapper::one());
void randomize_discrete (const Field& f, int seed,
                         const std::vector<ScalarWrapper>& values);
template<typename ST>
void randomize_discrete (const Field& f, int seed,
                         const std::vector<ST>& values)
{
  std::vector<ScalarWrapper> op_values;
  for (auto v : values)
    op_values.push_back(ScalarWrapper(v));
  return randomize_discrete(f,seed,op_values);
}

// Compute a random perturbation of a field for all view entries whose
// level index satisfies the mask.
// Input:
//   - f:             Field to perturbed. Required to have level midpoint
//                    tag as last dimension.
//   - perturb_level: magnitude of RELATIVE perturbation. That is,
//                    field *= (1 + U(1-perturb_level,1+perturb_level))
//   - base_seed:     Seed used for creating the engine input.
//   - level_mask:    A 1d int field (size of the level dimension of f) where f(i0,...,k) is
//                    perturbed if level_mask(k)=1
//   - dof_gids:      Field containing global DoF IDs for columns of f (if applicable)
void perturb (const Field& f,
              const ScalarWrapper perturb_level,
              const int base_seed,
              const Field& level_mask,
              const Field& dof_gids = Field());

// The reduction operator applied by vert/horiz contractions (see below)
enum class ContractionOp { Sum, Max, Min };

// Vertical/horizontal contractions of field.
// Computes a reduction of the weighted f_in along the contracted dimension:
//   f_out[j] = OP_i( weight[i] * f_in[i,j] )
// where OP is sum (the default), max, or min. Pass an all-ones weight for an
// unweighted reduction. If f_in has a valid mask (IntType, 0=invalid), only
// entries with nonzero mask participate.
// Output entries where NO input entry participates are set to the identity of
// OP: 0 for sum, and, for floating point types, -inf for max and +inf for min
// (the lowest/highest representable value for integer types). Callers that
// need a fill value there must detect those entries themselves (e.g., by
// contracting an all-ones field with the same mask, to count valid entries).
// Here, the weight is ASSUMED to be:
//  - a 1d (ncol) field for horiz_contraction
//  - either a 1d (nlev) or 2d (ncol,nlev) field for vert_contraction
// For horiz, the overload with comm also performs an all_reduce across ranks
void vert_contraction (const Field& f_out, const Field& f_in, const Field& weight,
                       const ContractionOp op = ContractionOp::Sum);
void horiz_contraction(const Field& f_out, const Field& f_in, const Field& weight,
                       const ContractionOp op = ContractionOp::Sum);
void horiz_contraction(const Field& f_out, const Field& f_in, const Field& weight, const ekat::Comm& comm,
                       const ContractionOp op = ContractionOp::Sum);

// Entrywise math functions, applied in place: f(i) = fn(f(i)), or
// f(i) = f(i)^p for apply_pow. If a mask (IntType, same layout as f) is
// given, only entries with nonzero mask are modified, which keeps fill
// values and other invalid entries out of the math (e.g., away from an FPE).
// Only floating point fields are supported.
// NOTE: these are not named sqrt/pow/etc, since a scream::pow would hide
//       ::pow for unqualified calls anywhere inside namespace scream.
enum class MathFn { Abs, Sqrt, Cbrt, Exp, Log, Log10, Sin, Cos, Tan };
void apply_math_fn (const Field& f, const MathFn fn);
void apply_math_fn (const Field& f, const MathFn fn, const Field& mask);
void apply_pow (const Field& f, const ScalarWrapper p);
void apply_pow (const Field& f, const ScalarWrapper p, const Field& mask);

// Reduce field to a single scalar, and return an opaque type, to allow hiding impl in cpp file.
// NOTE: all calculations are done serially HOST
ScalarWrapper frobenius_norm(const Field& f, const ekat::Comm* comm = nullptr);
ScalarWrapper inf_norm(const Field& f, const ekat::Comm* comm = nullptr);
ScalarWrapper field_sum(const Field& f, const ekat::Comm* comm = nullptr);
ScalarWrapper field_max(const Field& f, const ekat::Comm* comm = nullptr);
ScalarWrapper field_min(const Field& f, const ekat::Comm* comm = nullptr);

// Prints the value of a field at a certain location, specified by tags and indices.
// If the field layout contains all the location tags, we will slice the field along
// those tags, and print it. E.g., f might be a <COL,LEV> field, and the tags/indices
// refer to a single column, in which case we'll print a whole column worth of data.
void print_field_hyperslab (const Field& f,
                            const std::vector<FieldTag>& tags = {},
                            const std::vector<int>& indices = {},
                            std::ostream& out = std::cout);

// Compute where input field comparese correctly to given value
void compute_mask (const Field& x, const ScalarWrapper value, Comparison CMP, const Field& mask);
Field compute_mask (const Field& x, const ScalarWrapper value, Comparison CMP);
void compute_mask (const Field& lhs, const Field& rhs, Comparison CMP, const Field& mask);

// Transpose a field layout
void transpose (const Field& src, const Field& tgt);
Field transpose (const Field& src);

} // namespace scream

#endif // SCREAM_FIELD_UTILS_HPP
