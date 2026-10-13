#include "share/field/field_utils.hpp"
#include "share/field/field.hpp"

namespace scream {

namespace impl {

// The entrywise op: a math function, or a power with exponent p
struct MathOp {
  // Fn::Pow is not a MathFn, since only it needs the exponent
  enum class Fn { Abs, Sqrt, Cbrt, Exp, Log, Log10, Sin, Cos, Tan, Pow, Square };

  Fn     fn;
  double p = 0;

  template<typename ST>
  KOKKOS_INLINE_FUNCTION
  ST operator() (const ST x) const {
    switch (fn) {
      case Fn::Abs:    return Kokkos::abs(x);
      case Fn::Sqrt:   return Kokkos::sqrt(x);
      case Fn::Cbrt:   return Kokkos::cbrt(x);
      case Fn::Exp:    return Kokkos::exp(x);
      case Fn::Log:    return Kokkos::log(x);
      case Fn::Log10:  return Kokkos::log10(x);
      case Fn::Sin:    return Kokkos::sin(x);
      case Fn::Cos:    return Kokkos::cos(x);
      case Fn::Tan:    return Kokkos::tan(x);
      case Fn::Pow:    return Kokkos::pow(x,static_cast<ST>(p));
      case Fn::Square: return x*x;
    }
    // Silence NVCC warning "missing return statement at end of non-void function"
    return x;
  }
};

template<int N, typename ST>
void apply_math_op (const Field& f, const MathOp op, const Field* mask)
{
  using scalar_ND  = typename ekat::DataND<ST,N>::type;
  using int_ND     = typename ekat::DataND<const int,N>::type;
  using exec_space = typename Field::kt_dev::ExeSpace;
  using mdrange_t  = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<N>>;
  using range_t    = typename Field::kt_dev::RangePolicy;

  const auto& layout = f.get_header().get_identifier().get_layout();

  // Strided views work for any field (padded, subfield, ...)
  auto v = f.get_strided_view<scalar_ND>();
  const bool masked = mask!=nullptr;
  decltype(mask->get_strided_view<int_ND>()) m;
  if (masked) {
    m = mask->get_strided_view<int_ND>();
  }

  if constexpr (N==0) {
    Kokkos::parallel_for(range_t(0,1), KOKKOS_LAMBDA (int) {
      if (not masked or m()) v() = op(v());
    });
  } else if constexpr (N==1) {
    Kokkos::parallel_for(range_t(0,layout.dim(0)), KOKKOS_LAMBDA (int i) {
      if (not masked or m(i)) v(i) = op(v(i));
    });
  } else {
    int beg[N] = {};
    int end[N];
    for (int i=0; i<N; ++i) {
      end[i] = layout.dim(i);
    }
    mdrange_t policy(beg,end);
    if constexpr (N==2) {
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA (int i, int j) {
        if (not masked or m(i,j)) v(i,j) = op(v(i,j));
      });
    } else if constexpr (N==3) {
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA (int i, int j, int k) {
        if (not masked or m(i,j,k)) v(i,j,k) = op(v(i,j,k));
      });
    } else if constexpr (N==4) {
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA (int i, int j, int k, int l) {
        if (not masked or m(i,j,k,l)) v(i,j,k,l) = op(v(i,j,k,l));
      });
    } else if constexpr (N==5) {
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA (int i, int j, int k, int l, int n) {
        if (not masked or m(i,j,k,l,n)) v(i,j,k,l,n) = op(v(i,j,k,l,n));
      });
    } else if constexpr (N==6) {
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA (int i, int j, int k, int l, int n, int o) {
        if (not masked or m(i,j,k,l,n,o)) v(i,j,k,l,n,o) = op(v(i,j,k,l,n,o));
      });
    }
  }
}

template<typename ST>
void apply_math_op (const Field& f, const MathOp op, const Field* mask)
{
  switch (f.rank()) {
    case 0: apply_math_op<0,ST>(f,op,mask); break;
    case 1: apply_math_op<1,ST>(f,op,mask); break;
    case 2: apply_math_op<2,ST>(f,op,mask); break;
    case 3: apply_math_op<3,ST>(f,op,mask); break;
    case 4: apply_math_op<4,ST>(f,op,mask); break;
    case 5: apply_math_op<5,ST>(f,op,mask); break;
    case 6: apply_math_op<6,ST>(f,op,mask); break;
    default:
      EKAT_ERROR_MSG ("Error! Unsupported field rank in entrywise math function.\n"
                      " - field name: " + f.name() + "\n");
  }
}

void apply_math_op (const Field& f, const MathOp op, const Field* mask)
{
  EKAT_REQUIRE_MSG (f.is_allocated(),
      "Error! Cannot apply a math function to a field that is not yet allocated.\n"
      " - field name: " + f.name() + "\n");
  EKAT_REQUIRE_MSG (not f.is_read_only(),
      "Error! Cannot apply a math function to a read-only field.\n"
      " - field name: " + f.name() + "\n");
  if (mask) {
    EKAT_REQUIRE_MSG (mask->data_type()==DataType::IntType,
        "Error! The mask must have IntType data type.\n"
        " - mask name: " + mask->name() + "\n");
    EKAT_REQUIRE_MSG (mask->get_header().get_identifier().get_layout()==f.get_header().get_identifier().get_layout(),
        "Error! The mask must have the same layout as the field.\n"
        " - field name: " + f.name() + "\n"
        " - mask name : " + mask->name() + "\n");
  }

  switch (f.data_type()) {
    case DataType::FloatType:  apply_math_op<float >(f,op,mask); break;
    case DataType::DoubleType: apply_math_op<double>(f,op,mask); break;
    default:
      EKAT_ERROR_MSG ("Error! Entrywise math functions only support floating point fields.\n"
                      " - field name: " + f.name() + "\n"
                      " - data type : " + e2str(f.data_type()) + "\n");
  }
}

MathOp make_math_op (const MathFn fn)
{
  using Fn = MathOp::Fn;
  switch (fn) {
    case MathFn::Abs:   return {Fn::Abs};
    case MathFn::Sqrt:  return {Fn::Sqrt};
    case MathFn::Cbrt:  return {Fn::Cbrt};
    case MathFn::Exp:   return {Fn::Exp};
    case MathFn::Log:   return {Fn::Log};
    case MathFn::Log10: return {Fn::Log10};
    case MathFn::Sin:   return {Fn::Sin};
    case MathFn::Cos:   return {Fn::Cos};
    case MathFn::Tan:   return {Fn::Tan};
  }
  EKAT_ERROR_MSG ("Error! Unsupported MathFn.\n");
  return {Fn::Abs};
}

// Special-case the exponents with an exact (and cheaper) alternative,
// so that, e.g., x**2 is bit-for-bit x*x on every platform
MathOp make_pow_op (const double p)
{
  using Fn = MathOp::Fn;
  if (p==2) {
    return {Fn::Square};
  } else if (p==0.5) {
    return {Fn::Sqrt};
  }
  return {Fn::Pow,p};
}

} // namespace impl

void apply_math_fn (const Field& f, const MathFn fn)
{
  impl::apply_math_op(f,impl::make_math_op(fn),nullptr);
}

void apply_math_fn (const Field& f, const MathFn fn, const Field& mask)
{
  impl::apply_math_op(f,impl::make_math_op(fn),&mask);
}

void apply_pow (const Field& f, const ScalarWrapper p)
{
  if (p.value==1) return;
  impl::apply_math_op(f,impl::make_pow_op(p.value),nullptr);
}

void apply_pow (const Field& f, const ScalarWrapper p, const Field& mask)
{
  if (p.value==1) return;
  impl::apply_math_op(f,impl::make_pow_op(p.value),&mask);
}

} // namespace scream
