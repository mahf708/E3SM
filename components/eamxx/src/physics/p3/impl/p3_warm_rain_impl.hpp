#ifndef P3_WARM_RAIN_IMPL_HPP
#define P3_WARM_RAIN_IMPL_HPP

#include "p3_functions.hpp" // for ETI only but harmless for GPU

namespace scream {
namespace p3 {

/*
 * Implementation of the p3 warm-rain stage. Clients should NOT #include
 * this file, #include p3_functions.hpp instead.
 *
 * The warm-rain collision processes only depend on the in-cloud state after
 * the cloud/rain size distributions are computed, and none of the other part2
 * processes depend on them before back_to_cell_average. So part2's process loop
 * is split in three pointwise steps:
 *   1. size distributions (p3_main_size_distributions),
 *   2. warm-rain rates    (p3_main_warm_rain), stored in P3WarmRainRates,
 *   3. everything else    (p3_main_part2), which reads the stored rates.
 * With small kernels, each step is its own kernel, so the rates can be replaced
 * (e.g., by an emulator) between steps 2 and 3, outside of any kernel.
 */

template <typename S, typename D>
KOKKOS_FUNCTION
typename Functions<S,D>::Mask Functions<S,D>
::part2_skip_mask(
  const Int& k, const Int& nk, const Pack& qc, const Pack& qr, const Pack& qi,
  const Pack& T_atm, const Pack& qv_supersat_i)
{
  constexpr Scalar qsmall     = C::QSMALL;
  constexpr Scalar T_zerodegc = C::T_zerodegc.value;

  //compute mask to identify padded values in packs, which shouldn't be used in calculations
  const auto range_pack = ekat::range<IntPack>(k*Pack::n);
  const auto range_mask = range_pack < nk;

  // if relatively dry and no hydrometeors at this level, skip to end of k-loop (i.e. skip this level)
  return ( !range_mask ||
      (qc<qsmall && qr<qsmall && qi<qsmall &&
       T_atm<T_zerodegc && qv_supersat_i< -0.05) );
}

template <typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::part2_size_distributions(
  const view_dnu_table& dnu, const Pack& rho, const Pack& cld_frac_l, const Pack& cld_frac_r,
  const Pack& qc_incld, const Pack& qr_incld, const Pack& qi_incld, Pack& nc, Pack& nr,
  Pack& nc_incld, Pack& nr_incld, Pack& mu_c, Pack& nu, Pack& lamc, Pack& cdist, Pack& cdist1,
  Pack& mu_r, Pack& lamr, Pack& cdistr, Pack& logn0r, const P3Runtime& runtime_options,
  const Mask& not_skip_all)
{
  constexpr Scalar qsmall = C::QSMALL;
  constexpr Scalar nsmall = C::NSMALL;

  // skip micro process calculations except nucleation/acvtivation if there no hydrometeors are present
  const auto not_skip_micro = not_skip_all && (qc_incld >= qsmall || qr_incld >= qsmall || qi_incld >= qsmall);
  if (not not_skip_micro.any()) {
    return;
  }

  get_cloud_dsd2(qc_incld, nc_incld, mu_c, rho, nu, dnu,
                 lamc, cdist, cdist1, not_skip_micro);
  nc.set(not_skip_micro, nc_incld * cld_frac_l);

  get_rain_dsd2(qr_incld, nr_incld, mu_r, lamr, runtime_options, not_skip_micro);
  get_cdistr_logn0r(qr_incld, nr_incld, mu_r, lamr, cdistr, logn0r, not_skip_micro);
  nr.set(not_skip_micro, nr_incld * cld_frac_r);

  // impose lower limit to prevent taking log of # < 0 in the ice-rain lookup tables
  // (it also affects rain self-collection, which comes before them)
  const auto qi_gt_small = qi_incld >= qsmall && not_skip_micro;
  nr_incld.set(qi_gt_small, max(nr_incld, nsmall));
}

template <typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::warm_rain_processes(
  const Pack& rho, const Pack& inv_rho, const Pack& qc_incld, const Pack& nc_incld,
  const Pack& qr_incld, const Pack& nr_incld, const Pack& inv_qc_relvar, const Pack& mu_c,
  const Pack& nu, Pack& qc2qr_autoconv_tend, Pack& nc2nr_autoconv_tend, Pack& ncautr,
  Pack& nc_selfcollect_tend, Pack& qc2qr_accret_tend, Pack& nc_accret_tend,
  Pack& nr_selfcollect_tend, const P3Runtime& runtime_options, const Mask& context)
{
  qc2qr_autoconv_tend = 0;
  nc2nr_autoconv_tend = 0;
  ncautr              = 0;
  nc_selfcollect_tend = 0;
  qc2qr_accret_tend   = 0;
  nc_accret_tend      = 0;
  nr_selfcollect_tend = 0;

  // cloud water autoconversion
  // NOTE: cloud_water_autoconversion must be called before droplet_self_collection
  cloud_water_autoconversion(
    rho, qc_incld, nc_incld, inv_qc_relvar,
    qc2qr_autoconv_tend, nc2nr_autoconv_tend, ncautr, runtime_options, context);

  // self-collection of droplets
  droplet_self_collection(
    rho, inv_rho, qc_incld,
    mu_c, nu, nc2nr_autoconv_tend, nc_selfcollect_tend, context);

  // accretion of cloud by rain
  cloud_rain_accretion(
    rho, inv_rho, qc_incld, nc_incld, qr_incld, inv_qc_relvar,
    qc2qr_accret_tend, nc_accret_tend, runtime_options, context);

  // self-collection and breakup of rain
  // (breakup following modified Verlinde and Cotton scheme)
  rain_self_collection(
    rho, qr_incld, nr_incld,
    nr_selfcollect_tend, runtime_options, context);
}

template <typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::warm_rain_emulator_merge(
  const Pack& cld_frac_l, const Pack& cld_frac_r, const Pack& nc, const Pack& nr,
  const Scalar& inv_dt, const Scalar& kk_factor, const bool& do_cloud_self_collection,
  const Pack& emu_qc2qr_autoconv_tend, const Pack& emu_nc2nr_autoconv_tend,
  const Pack& emu_ncautr, const Pack& emu_nc_selfcollect_tend,
  const Pack& emu_qc2qr_accret_tend, const Pack& emu_nc_accret_tend,
  const Pack& emu_nr_selfcollect_tend, const Pack& emu_use_cloud, const Pack& emu_use_rain,
  Pack& qc2qr_autoconv_tend, Pack& nc2nr_autoconv_tend, Pack& ncautr,
  Pack& nc_selfcollect_tend, Pack& qc2qr_accret_tend, Pack& nc_accret_tend,
  Pack& nr_selfcollect_tend, const Mask& context)
{
  const auto use_cloud = emu_use_cloud > sp(0.5) && context;
  const auto use_rain  = emu_use_rain  > sp(0.5) && context;
  const auto fallback  = !use_cloud && context;

  // Cloud fractions used by back_to_cell_average for each warm-rain rate
  const Pack lr_cldm = min(cld_frac_l, cld_frac_r);

  // Outside the emulator's envelope: stock P3, with scaled autoconversion
  qc2qr_autoconv_tend.set(fallback, qc2qr_autoconv_tend * kk_factor);
  nc2nr_autoconv_tend.set(fallback, nc2nr_autoconv_tend * kk_factor);
  ncautr.set(fallback, ncautr * kk_factor);

  if (use_cloud.any()) {
    // Cloud self-collection is zero in stock P3, so it can be turned off to isolate its effect
    const Pack emu_nc_sc = do_cloud_self_collection ? emu_nc_selfcollect_tend : Pack(0);

    // nc_conservation uses nc + nc_selfcollect_tend*dt as the source for the other
    // nc sinks, so it must stay non-negative
    const Pack nc_sc = max(emu_nc_sc, -max(nc, 0) * inv_dt);

    qc2qr_autoconv_tend.set(use_cloud, emu_qc2qr_autoconv_tend / cld_frac_l);
    nc2nr_autoconv_tend.set(use_cloud, emu_nc2nr_autoconv_tend / cld_frac_l);
    ncautr.set(use_cloud, emu_ncautr / lr_cldm);
    nc_selfcollect_tend.set(use_cloud, nc_sc / cld_frac_l);
    qc2qr_accret_tend.set(use_cloud, emu_qc2qr_accret_tend / lr_cldm);
    nc_accret_tend.set(use_cloud, emu_nc_accret_tend / lr_cldm);
  }

  if (use_rain.any()) {
    // nr_conservation counts nc2nr_autoconv_tend (not ncautr) as the rain-number
    // source from autoconversion, so its check alone is too lenient here
    const Pack nr_sc = min(emu_nr_selfcollect_tend, max(nr, 0) * inv_dt + ncautr * lr_cldm);
    nr_selfcollect_tend.set(use_rain, nr_sc / cld_frac_r);
  }
}

template <typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::p3_main_size_distributions(
  const MemberType& team, const Int& nk_pack, const Int& nk, const view_dnu_table& dnu,
  const uview_1d<const Pack>& cld_frac_l, const uview_1d<const Pack>& cld_frac_r,
  const uview_1d<const Pack>& qc, const uview_1d<const Pack>& qr,
  const uview_1d<const Pack>& qi, const uview_1d<const Pack>& T_atm,
  const uview_1d<const Pack>& qv_supersat_i, const uview_1d<const Pack>& rho,
  const uview_1d<const Pack>& qc_incld, const uview_1d<const Pack>& qr_incld,
  const uview_1d<const Pack>& qi_incld, const uview_1d<Pack>& nc, const uview_1d<Pack>& nr,
  const uview_1d<Pack>& nc_incld, const uview_1d<Pack>& nr_incld, const uview_1d<Pack>& mu_c,
  const uview_1d<Pack>& nu, const uview_1d<Pack>& lamc, const uview_1d<Pack>& cdist,
  const uview_1d<Pack>& cdist1, const uview_1d<Pack>& mu_r, const uview_1d<Pack>& lamr,
  const uview_1d<Pack>& cdistr, const uview_1d<Pack>& logn0r,
  const P3Runtime& runtime_options)
{
  Kokkos::parallel_for(
    Kokkos::TeamVectorRange(team, nk_pack), [&] (Int k) {

    const auto skip_all = part2_skip_mask(k, nk, qc(k), qr(k), qi(k), T_atm(k), qv_supersat_i(k));
    if (skip_all.all()) {
      return;
    }

    part2_size_distributions(
      dnu, rho(k), cld_frac_l(k), cld_frac_r(k), qc_incld(k), qr_incld(k), qi_incld(k),
      nc(k), nr(k), nc_incld(k), nr_incld(k), mu_c(k), nu(k), lamc(k), cdist(k), cdist1(k),
      mu_r(k), lamr(k), cdistr(k), logn0r(k), runtime_options, !skip_all);
  });
  team.team_barrier();
}

template <typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::p3_main_warm_rain(
  const MemberType& team, const Int& nk_pack, const Int& nk,
  const uview_1d<const Pack>& inv_qc_relvar, const uview_1d<const Pack>& qc,
  const uview_1d<const Pack>& qr, const uview_1d<const Pack>& qi,
  const uview_1d<const Pack>& T_atm, const uview_1d<const Pack>& qv_supersat_i,
  const uview_1d<const Pack>& rho, const uview_1d<const Pack>& inv_rho,
  const uview_1d<const Pack>& qc_incld, const uview_1d<const Pack>& nc_incld,
  const uview_1d<const Pack>& qr_incld, const uview_1d<const Pack>& nr_incld,
  const uview_1d<const Pack>& mu_c, const uview_1d<const Pack>& nu,
  const P3WarmRainRates1d& warm_rain, const P3Runtime& runtime_options)
{
  Kokkos::parallel_for(
    Kokkos::TeamVectorRange(team, nk_pack), [&] (Int k) {

    const auto skip_all = part2_skip_mask(k, nk, qc(k), qr(k), qi(k), T_atm(k), qv_supersat_i(k));

    warm_rain_processes(
      rho(k), inv_rho(k), qc_incld(k), nc_incld(k), qr_incld(k), nr_incld(k),
      inv_qc_relvar(k), mu_c(k), nu(k),
      warm_rain.qc2qr_autoconv_tend(k), warm_rain.nc2nr_autoconv_tend(k), warm_rain.ncautr(k),
      warm_rain.nc_selfcollect_tend(k), warm_rain.qc2qr_accret_tend(k),
      warm_rain.nc_accret_tend(k), warm_rain.nr_selfcollect_tend(k),
      runtime_options, !skip_all);
  });
  team.team_barrier();
}

} // namespace p3
} // namespace scream

#endif // P3_WARM_RAIN_IMPL_HPP
