#include "share/field/field_utils.hpp"

#include <Kokkos_Random.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace scream {

namespace impl {

// SplitMix64 finalizer: cheap, well-mixed 64-bit hash
KOKKOS_INLINE_FUNCTION
std::uint64_t splitmix64 (std::uint64_t x)
{
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Access a rank-N view with an array of N indices
template<typename ViewT, typename IdxT, std::size_t... Is>
KOKKOS_INLINE_FUNCTION
typename ViewT::reference_type
entry (const ViewT& v, const IdxT& idx, std::index_sequence<Is...>)
{
  return v(idx[Is]...);
}

template<int N, typename ST>
void perturb (const Field& f, const ST perturb_level,
              const int base_seed,
              const Field& level_mask,
              const Field& dof_gids)
{
  using namespace ShortFieldTagsNames;
  using data_t     = typename ekat::DataND<ST,N>::type;
  using exec_space = typename Field::kt_dev::ExeSpace;
  using gen_t      = Kokkos::Random_XorShift64<exec_space>;

  const auto& fl = f.get_header().get_identifier().get_layout();
  const bool has_col = fl.has_tag(COL);
  const bool has_lev = fl.has_tag(LEV);

  auto v = f.get_strided_view<data_t>();
  Field::strided_view_dev_t<const int*> lev_mask, gids;
  if (has_lev)
    lev_mask = level_mask.get_strided_view<const int*>();
  if (has_col)
    gids = dof_gids.get_strided_view<const int*>();

  Kokkos::Array<int,N> dims;
  for (int d=0; d<N; ++d) {
    dims[d] = fl.dim(d);
  }

  const ST lb = 1 - perturb_level;
  // Kokkos::rand draws ints in [start,end), while we want [lb,ub] for ints
  const ST ub = std::is_integral_v<ST> ? 2 + perturb_level : 1 + perturb_level;
  const std::uint64_t seed_hash = splitmix64(static_cast<std::uint64_t>(static_cast<std::int64_t>(base_seed)));
  const int first_inner = has_col ? 1 : 0;

  // Each entry gets its own generator, seeded by hashing (base_seed, gid, ilev, inner),
  // where inner is the flattened index over the dims between COL and LEV. Hence, the
  // value applied to an entry does not depend on thread scheduling or on how columns
  // are distributed across MPI ranks.
  auto lambda = KOKKOS_LAMBDA (const int flat) {
    Kokkos::Array<int,N> idx;
    int r = flat;
    for (int d=N-1; d>=0; --d) {
      idx[d] = r % dims[d];
      r /= dims[d];
    }

    const int ilev = idx[N-1];
    if (has_lev and lev_mask(ilev)==0) return;

    const std::int64_t gid = has_col ? gids(idx[0]) : 0;
    int inner = 0;
    if constexpr (N>1) {
      for (int d=first_inner; d<N-1; ++d) {
        inner = inner*dims[d] + idx[d];
      }
    }

    std::uint64_t h = splitmix64(seed_hash ^ static_cast<std::uint64_t>(gid));
    h = splitmix64(h ^ static_cast<std::uint64_t>(ilev));
    h = splitmix64(h ^ static_cast<std::uint64_t>(inner));

    gen_t gen(h);
    entry(v,idx,std::make_index_sequence<N>{}) *= Kokkos::rand<gen_t,ST>::draw(gen,lb,ub);
  };
  Kokkos::parallel_for("perturb", Kokkos::RangePolicy<exec_space>(0,static_cast<int>(fl.size())), lambda);
}

template<typename ST>
void perturb (const Field& f, const ST perturb_level,
              const int base_seed,
              const Field& level_mask,
              const Field& dof_gids)
{
  switch (f.rank()) {
    case 1: perturb<1>(f,perturb_level,base_seed,level_mask,dof_gids); break;
    case 2: perturb<2>(f,perturb_level,base_seed,level_mask,dof_gids); break;
    case 3: perturb<3>(f,perturb_level,base_seed,level_mask,dof_gids); break;
    case 4: perturb<4>(f,perturb_level,base_seed,level_mask,dof_gids); break;
    case 5: perturb<5>(f,perturb_level,base_seed,level_mask,dof_gids); break;
    case 6: perturb<6>(f,perturb_level,base_seed,level_mask,dof_gids); break;
    default:
      EKAT_ERROR_MSG ("[perturb] Error! Unsupported field rank.\n"
                      " - field name: " + f.name() + "\n"
                      " - field rank: " + std::to_string(f.rank()) + "\n");
  }
}

} // namespace impl

void perturb (const Field& f, const ScalarWrapper perturb_level,
              const int base_seed,
              const Field& level_mask,
              const Field& dof_gids)
{
  using namespace ShortFieldTagsNames;

  // Perform some sanity checks, before handing off to the impl method
  EKAT_REQUIRE_MSG(f.is_allocated(),
    "[perturb] Error! Trying to perturb the values of a field not yet allocated.\n"
    " - field name: " + f.name() + "\n");

  const auto& fl = f.get_header().get_identifier().get_layout();

  // Field we are perturbing should have a level dimension,
  // and it is required to be the last dimension
  EKAT_REQUIRE_MSG(fl.rank()>0 && (fl.tags().back() == LEV || fl.tags().back() == ILEV),
    "[perturb] Error! Field must have LEV or ILEV as its last dimension.\n"
    " - field name: " + f.name() + "\n"
    " - field layout: " + fl.to_string() + "\n");

  if (fl.has_tag(COL)) {
    // If field has a column dimension, it should be the first dimension
    EKAT_REQUIRE_MSG(fl.tag(0) == COL,
      "[perturb] Error! Fields with COL dimension should have it as their first dimension.\n"
      " - field name: " + f.name() + "\n"
      " - field layout: " + fl.to_string() + "\n");

    const auto& dof_gids_fl = dof_gids.get_header().get_identifier().get_layout();
    EKAT_REQUIRE_MSG(dof_gids_fl.dim(0) == fl.dim(COL),
      "Error! Field of DoF GIDs should have the same size as perturbed field's column dimension.\n"
      " - field name: " + f.name() + "\n"
      " - field layout: " + fl.to_string() + "\n"
      " - dof_gids layout: " + dof_gids_fl.to_string() + "\n");
    EKAT_REQUIRE_MSG(dof_gids.data_type() == DataType::IntType,
      "Error! DoF GIDs field must have \"int\" as data type.\n"
      " - dof gids data type: " + e2str(dof_gids.data_type()));
  }

  switch (f.data_type()) {
    case DataType::IntType:
      impl::perturb(f,perturb_level.as<int>(),base_seed,level_mask,dof_gids);
      break;
    case DataType::FloatType:
      impl::perturb(f,perturb_level.as<float>(),base_seed,level_mask,dof_gids);
      break;
    case DataType::DoubleType:
      impl::perturb(f,perturb_level.as<double>(),base_seed,level_mask,dof_gids);
      break;
    default:
      EKAT_ERROR_MSG ("[perturb] Error! Unsupported field data type.\n");
  }
}

} // namespace scream
