#include "p3_functions.hpp" // for ETI only but harmless for GPU

namespace scream {
namespace p3 {

/*
 * Small-kernel drivers of the p3 warm-rain stage (see p3_warm_rain_impl.hpp).
 * All kernels are fully parallel over (column, level pack).
 */

template <>
void Functions<Real,DefaultDevice>
::p3_main_size_distributions_disp(
  const Int& nj, const Int& nk, const view_dnu_table& dnu,
  const uview_2d<const Pack>& cld_frac_l, const uview_2d<const Pack>& cld_frac_r,
  const uview_2d<const Pack>& qc, const uview_2d<const Pack>& qr,
  const uview_2d<const Pack>& qi, const uview_2d<const Pack>& T_atm,
  const uview_2d<const Pack>& qv_supersat_i, const uview_2d<const Pack>& rho,
  const uview_2d<const Pack>& qc_incld, const uview_2d<const Pack>& qr_incld,
  const uview_2d<const Pack>& qi_incld, const uview_2d<Pack>& nc, const uview_2d<Pack>& nr,
  const uview_2d<Pack>& nc_incld, const uview_2d<Pack>& nr_incld, const uview_2d<Pack>& mu_c,
  const uview_2d<Pack>& nu, const uview_2d<Pack>& lamc, const uview_2d<Pack>& cdist,
  const uview_2d<Pack>& cdist1, const uview_2d<Pack>& mu_r, const uview_2d<Pack>& lamr,
  const uview_2d<Pack>& cdistr, const uview_2d<Pack>& logn0r,
  const uview_1d<bool>& nucleationPossible, const uview_1d<bool>& hydrometeorsPresent,
  const P3Runtime& runtime_options)
{
  using ExeSpace = typename KT::ExeSpace;

  const Int nk_pack = ekat::npack<Pack>(nk);
  Kokkos::parallel_for(
    "p3_main_size_distributions_disp",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<2>>({0, 0}, {nj, nk_pack}),
    KOKKOS_LAMBDA (const int i, const int k) {

    // Same column skip as part2
    if (!(nucleationPossible(i) || hydrometeorsPresent(i))) {
      return;
    }

    const auto skip_all = part2_skip_mask(k, nk, qc(i,k), qr(i,k), qi(i,k), T_atm(i,k), qv_supersat_i(i,k));
    if (skip_all.all()) {
      return;
    }

    part2_size_distributions(
      dnu, rho(i,k), cld_frac_l(i,k), cld_frac_r(i,k), qc_incld(i,k), qr_incld(i,k), qi_incld(i,k),
      nc(i,k), nr(i,k), nc_incld(i,k), nr_incld(i,k), mu_c(i,k), nu(i,k), lamc(i,k), cdist(i,k),
      cdist1(i,k), mu_r(i,k), lamr(i,k), cdistr(i,k), logn0r(i,k), runtime_options, !skip_all);
  });
}

template <>
void Functions<Real,DefaultDevice>
::p3_main_warm_rain_disp(
  const Int& nj, const Int& nk, const uview_2d<const Pack>& inv_qc_relvar,
  const uview_2d<const Pack>& qc, const uview_2d<const Pack>& qr,
  const uview_2d<const Pack>& qi, const uview_2d<const Pack>& T_atm,
  const uview_2d<const Pack>& qv_supersat_i, const uview_2d<const Pack>& rho,
  const uview_2d<const Pack>& inv_rho, const uview_2d<const Pack>& qc_incld,
  const uview_2d<const Pack>& nc_incld, const uview_2d<const Pack>& qr_incld,
  const uview_2d<const Pack>& nr_incld, const uview_2d<const Pack>& mu_c,
  const uview_2d<const Pack>& nu, const P3WarmRainRates2d& warm_rain,
  const uview_1d<bool>& nucleationPossible, const uview_1d<bool>& hydrometeorsPresent,
  const P3Runtime& runtime_options)
{
  using ExeSpace = typename KT::ExeSpace;

  const Int nk_pack = ekat::npack<Pack>(nk);
  Kokkos::parallel_for(
    "p3_main_warm_rain_disp",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<2>>({0, 0}, {nj, nk_pack}),
    KOKKOS_LAMBDA (const int i, const int k) {

    // Rates are always defined (zero where part2 does not run)
    const bool active_col = nucleationPossible(i) || hydrometeorsPresent(i);
    const auto skip_all = !Mask(active_col) ||
      part2_skip_mask(k, nk, qc(i,k), qr(i,k), qi(i,k), T_atm(i,k), qv_supersat_i(i,k));

    warm_rain_processes(
      rho(i,k), inv_rho(i,k), qc_incld(i,k), nc_incld(i,k), qr_incld(i,k), nr_incld(i,k),
      inv_qc_relvar(i,k), mu_c(i,k), nu(i,k),
      warm_rain.qc2qr_autoconv_tend(i,k), warm_rain.nc2nr_autoconv_tend(i,k),
      warm_rain.ncautr(i,k), warm_rain.nc_selfcollect_tend(i,k),
      warm_rain.qc2qr_accret_tend(i,k), warm_rain.nc_accret_tend(i,k),
      warm_rain.nr_selfcollect_tend(i,k), runtime_options, !skip_all);
  });
}

template <>
void Functions<Real,DefaultDevice>
::warm_rain_emulator_inputs_disp(
  const Int& nj, const Int& nk, const P3PrognosticState& prognostic_state,
  const P3Temporaries& temporaries, const uview_2d<Pack>& emu_qc,
  const uview_2d<Pack>& emu_nc, const uview_2d<Pack>& emu_qr, const uview_2d<Pack>& emu_nr,
  const uview_2d<Pack>& emu_rho)
{
  using ExeSpace = typename KT::ExeSpace;

  const auto qc  = prognostic_state.qc;
  const auto nc  = prognostic_state.nc;
  const auto qr  = prognostic_state.qr;
  const auto nr  = prognostic_state.nr;
  const auto rho = temporaries.rho;

  const Int nk_pack = ekat::npack<Pack>(nk);
  Kokkos::parallel_for(
    "p3_warm_rain_emulator_inputs_disp",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<2>>({0, 0}, {nj, nk_pack}),
    KOKKOS_LAMBDA (const int i, const int k) {
    emu_qc(i,k)  = qc(i,k);
    emu_nc(i,k)  = nc(i,k);
    emu_qr(i,k)  = qr(i,k);
    emu_nr(i,k)  = nr(i,k);
    emu_rho(i,k) = rho(i,k);
  });
}

template <>
void Functions<Real,DefaultDevice>
::warm_rain_emulator_merge_disp(
  const Int& nj, const Int& nk, const Scalar& dt, const Scalar& kk_factor,
  const bool& do_cloud_self_collection, const P3PrognosticState& prognostic_state,
  const P3DiagnosticInputs& diagnostic_inputs, const P3Temporaries& temporaries,
  const P3WarmRainRates<uview_2d<const Pack>>& emu,
  const uview_2d<const Pack>& emu_use_cloud, const uview_2d<const Pack>& emu_use_rain)
{
  using ExeSpace = typename KT::ExeSpace;

  const auto nc         = prognostic_state.nc;
  const auto nr         = prognostic_state.nr;
  const auto cld_frac_l = diagnostic_inputs.cld_frac_l;
  const auto cld_frac_r = diagnostic_inputs.cld_frac_r;
  const auto warm_rain  = temporaries.warm_rain;
  const Scalar inv_dt   = 1 / dt;

  const Int nk_pack = ekat::npack<Pack>(nk);
  Kokkos::parallel_for(
    "p3_warm_rain_emulator_merge_disp",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<2>>({0, 0}, {nj, nk_pack}),
    KOKKOS_LAMBDA (const int i, const int k) {

    // Padding is skipped. Levels/columns skipped by part2 are harmless: part2 ignores them.
    const auto range_mask = ekat::range<IntPack>(k*Pack::n) < nk;

    warm_rain_emulator_merge(
      cld_frac_l(i,k), cld_frac_r(i,k), nc(i,k), nr(i,k), inv_dt, kk_factor,
      do_cloud_self_collection,
      emu.qc2qr_autoconv_tend(i,k), emu.nc2nr_autoconv_tend(i,k), emu.ncautr(i,k),
      emu.nc_selfcollect_tend(i,k), emu.qc2qr_accret_tend(i,k), emu.nc_accret_tend(i,k),
      emu.nr_selfcollect_tend(i,k), emu_use_cloud(i,k), emu_use_rain(i,k),
      warm_rain.qc2qr_autoconv_tend(i,k), warm_rain.nc2nr_autoconv_tend(i,k),
      warm_rain.ncautr(i,k), warm_rain.nc_selfcollect_tend(i,k),
      warm_rain.qc2qr_accret_tend(i,k), warm_rain.nc_accret_tend(i,k),
      warm_rain.nr_selfcollect_tend(i,k), range_mask);
  });
}

} // namespace p3
} // namespace scream
