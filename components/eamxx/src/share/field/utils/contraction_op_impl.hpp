#ifndef EAMXX_CONTRACTION_OP_IMPL_HPP
#define EAMXX_CONTRACTION_OP_IMPL_HPP

#include "share/field/field_utils.hpp"

#include <ekat_assert.hpp>

#include <mpi.h>

#include <type_traits>

namespace scream {
namespace impl {

// Compile-time traits of a ContractionOp, shared by the vert/horiz kernels:
//  - reducer: the Kokkos reducer combining partial results across threads
//  - join:    folds one (weighted) entry into a partial result
template<typename ST, ContractionOp OP>
struct ContractionOpTraits {
  using reducer = std::conditional_t<OP==ContractionOp::Sum, Kokkos::Sum<ST>,
                  std::conditional_t<OP==ContractionOp::Max, Kokkos::Max<ST>,
                                                             Kokkos::Min<ST>>>;

  KOKKOS_INLINE_FUNCTION
  static void join (ST& acc, const ST v) {
    if constexpr (OP==ContractionOp::Sum) {
      acc += v;
    } else if constexpr (OP==ContractionOp::Max) {
      acc = v>acc ? v : acc;
    } else {
      acc = v<acc ? v : acc;
    }
  }
};

inline MPI_Op mpi_op (const ContractionOp op) {
  switch (op) {
    case ContractionOp::Sum: return MPI_SUM;
    case ContractionOp::Max: return MPI_MAX;
    case ContractionOp::Min: return MPI_MIN;
  }
  EKAT_ERROR_MSG ("Error! Unsupported ContractionOp.\n");
  return MPI_OP_NULL;
}

} // namespace impl
} // namespace scream

#endif // EAMXX_CONTRACTION_OP_IMPL_HPP
