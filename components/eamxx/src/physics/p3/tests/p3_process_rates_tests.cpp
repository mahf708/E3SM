#include "catch2/catch.hpp"

#include "p3_functions.hpp"
#include "p3_process_rates.hpp"
#include "p3_test_data.hpp"
#include "p3_unit_tests_common.hpp"

#include "share/core/eamxx_types.hpp"

#include <set>
#include <vector>
#include <string>

namespace scream {
namespace p3 {
namespace unit_test {

template <typename D>
struct UnitWrap::UnitTest<D>::TestP3ProcessRates : public UnitWrap::UnitTest<D>::Base {

  void run_registry ()
  {
    using PR = P3ProcessRates;

    std::set<std::string> names;
    for (int i = 0; i < PR::num_rates; ++i) {
      names.insert(PR::name(i));
      REQUIRE (PR::index(PR::name(i)) == i);
    }
    REQUIRE (names.size() == static_cast<size_t>(PR::num_rates));
    REQUIRE (PR::index("not_a_rate") == -1);
    REQUIRE (PR::num_packs == PR::num_rates-1);
    REQUIRE (std::string(PR::name(PR::wetgrowth)) == "wetgrowth");
  }

#ifdef SCREAM_P3_SMALL_KERNELS
  // With a hook, part2 computes the rates, runs the hook, then applies the rates
  void run_hook ()
  {
    using P3F  = Functions;
    using PR   = P3ProcessRates;
    using Hook = typename P3F::P3ProcessRatesHook;
    using Hooks = typename P3F::P3Hooks;

    auto engine = Base::get_engine();

    // Warm, ice-free columns with no rain: rain can only come from the warm-rain processes
    //               its, ite, kts, kte, it,  dt, do_predict_nc, do_prescribed_CCN
    P3MainData d_ref(1,   8,   1,  72,  1, 300, true,          false);
    d_ref.randomize(engine, {
        {d_ref.pres           , {5.0e+04, 1.0e+05}},
        {d_ref.dz             , {1.0e+02, 3.0e+02}},
        {d_ref.nc_nuceat_tend , {0      , 0}},
        {d_ref.nccn_prescribed, {0      , 0}},
        {d_ref.ni_activated   , {0      , 0}},
        {d_ref.dpres          , {1.0e+03, 1.5e+03}},
        {d_ref.inv_exner      , {1      , 1}},
        {d_ref.cld_frac_i     , {1      , 1}},
        {d_ref.cld_frac_l     , {0.2    , 1}},
        {d_ref.cld_frac_r     , {0.2    , 1}},
        {d_ref.inv_qc_relvar  , {1      , 1}},
        {d_ref.qc             , {1.0e-5 , 1.0e-3}},
        {d_ref.nc             , {1.0e+07, 1.0e+08}},
        {d_ref.qr             , {0      , 0}},
        {d_ref.nr             , {0      , 0}},
        {d_ref.qi             , {0      , 0}},
        {d_ref.qm             , {0      , 0}},
        {d_ref.ni             , {0      , 0}},
        {d_ref.bm             , {0      , 0}},
        {d_ref.qv             , {5.0e-3 , 1.0e-2}},
        {d_ref.qv_prev        , {5.0e-3 , 1.0e-2}},
        {d_ref.th_atm         , {2.85e+02, 3.0e+02}},
        {d_ref.t_prev         , {2.85e+02, 3.0e+02}}
    });
    P3MainData d_noop(d_ref), d_zero(d_ref);

    auto run = [](P3MainData& d, const Hook& hook) {
      Hooks hooks;
      hooks.process_rates = hook;
      p3_main_host_hook(
        d.qc, d.nc, d.qr, d.nr, d.th_atm, d.qv, d.dt, d.qi, d.qm, d.ni,
        d.bm, d.pres, d.dz, d.nc_nuceat_tend, d.nccn_prescribed, d.ni_activated, d.inv_qc_relvar, d.it, d.precip_liq_surf,
        d.precip_ice_surf, d.its, d.ite, d.kts, d.kte, d.diag_eff_radius_qc, d.diag_eff_radius_qi, d.diag_eff_radius_qr,
        d.rho_qi, d.do_predict_nc, d.do_prescribed_CCN, d.use_hetfrz_classnuc, d.dpres, d.inv_exner, d.qv2qi_depos_tend,
        d.precip_liq_flux, d.precip_ice_flux, d.cld_frac_r, d.cld_frac_l, d.cld_frac_i,
        d.liq_ice_exchange, d.vap_liq_exchange, d.vap_ice_exchange, d.qv_prev, d.t_prev, hooks);
    };

    const Int ncol = d_ref.ite - d_ref.its + 1, nk = d_ref.kte - d_ref.kts + 1;
    auto storage = [&]() {
      return typename P3F::template view_3d<Pack>("process_rates", ncol, PR::num_rates, ekat::npack<Pack>(nk));
    };

    int num_calls = 0;
    run(d_ref, Hook());

    // A hook that changes nothing changes nothing
    Hook noop;
    noop.process_rates = storage();
    noop.callback = [&](const typename P3F::P3ProcessState& s) {
      ++num_calls;
      REQUIRE (s.ncol == ncol);
      REQUIRE (s.nlev == nk);
      REQUIRE (s.state.count("qc") == 1);
      REQUIRE (s.state.count("rho") == 1);
      REQUIRE (s.state.count("nc_incld") == 1);
    };
    run(d_noop, noop);

    // Turn the warm-rain collision processes off, by name
    Hook zero;
    zero.process_rates = storage();
    zero.callback = [&](const typename P3F::P3ProcessState& s) {
      ++num_calls;
      for (const auto name : {"qc2qr_autoconv_tend", "nc2nr_autoconv_tend", "ncautr", "nc_selfcollect_tend",
                              "qc2qr_accret_tend", "nc_accret_tend", "nr_selfcollect_tend"}) {
        const int r = PR::index(name);
        REQUIRE (r >= 0);
        Kokkos::deep_copy(Kokkos::subview(s.process_rates, Kokkos::ALL, r, Kokkos::ALL), Pack(0));
      }
    };
    run(d_zero, zero);
    REQUIRE (num_calls == 2);

    const auto tot = d_ref.total(d_ref.qc);
    for (Int t = 0; t < tot; ++t) {
      REQUIRE(d_noop.qc[t] == d_ref.qc[t]);
      REQUIRE(d_noop.nc[t] == d_ref.nc[t]);
      REQUIRE(d_noop.qr[t] == d_ref.qr[t]);
      REQUIRE(d_noop.nr[t] == d_ref.nr[t]);
      REQUIRE(d_noop.qv[t] == d_ref.qv[t]);
      REQUIRE(d_noop.th_atm[t] == d_ref.th_atm[t]);
    }

    Real qr_max_ref = 0;
    for (Int t = 0; t < tot; ++t) {
      qr_max_ref = std::max(qr_max_ref, d_ref.qr[t]);
      REQUIRE(d_zero.qr[t] == 0);
    }
    REQUIRE (qr_max_ref > 0);
  }

  // In columns where nothing happens, part2 skips its work: the stored rates
  // must be zero, not what the storage held before
  void run_quiet_columns ()
  {
    using P3F   = Functions;
    using PR    = P3ProcessRates;
    using Hook  = typename P3F::P3ProcessRatesHook;
    using Hooks = typename P3F::P3Hooks;

    auto engine = Base::get_engine();

    // Warm, dry columns with no hydrometeors: no nucleation, nothing to do
    //           its, ite, kts, kte, it,  dt, do_predict_nc, do_prescribed_CCN
    P3MainData d(1,   4,   1,  20,  1, 300, true,          false);
    d.randomize(engine, {
        {d.pres, {8.0e+04, 1.0e+05}}, {d.dz, {1.0e+02, 3.0e+02}}, {d.dpres, {1.0e+03, 1.5e+03}},
        {d.nc_nuceat_tend, {0, 0}}, {d.nccn_prescribed, {0, 0}}, {d.ni_activated, {0, 0}},
        {d.inv_exner, {1, 1}}, {d.cld_frac_i, {1, 1}}, {d.cld_frac_l, {1, 1}}, {d.cld_frac_r, {1, 1}},
        {d.inv_qc_relvar, {1, 1}},
        {d.qc, {0, 0}}, {d.nc, {0, 0}}, {d.qr, {0, 0}}, {d.nr, {0, 0}},
        {d.qi, {0, 0}}, {d.qm, {0, 0}}, {d.ni, {0, 0}}, {d.bm, {0, 0}},
        {d.qv, {1.0e-4, 2.0e-4}}, {d.qv_prev, {1.0e-4, 2.0e-4}},
        {d.th_atm, {2.9e+02, 3.0e+02}}, {d.t_prev, {2.9e+02, 3.0e+02}}
    });

    const Int ncol = d.ite - d.its + 1, nk = d.kte - d.kts + 1;
    Hook hook;
    hook.process_rates = typename P3F::template view_3d<Pack>("process_rates", ncol, PR::num_rates, ekat::npack<Pack>(nk));
    Kokkos::deep_copy(hook.process_rates, Pack(7)); // last step's leftovers
    int num_calls = 0;
    hook.callback = [&](const typename P3F::P3ProcessState& s) {
      ++num_calls;
      auto r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.process_rates);
      for (int i = 0; i < ncol; ++i)
        for (int n = 0; n < PR::num_rates; ++n)
          for (int k = 0; k < nk; ++k)
            REQUIRE (r(i, n, k / Pack::n)[k % Pack::n] == 0);
    };
    Hooks hooks;
    hooks.process_rates = hook;
    p3_main_host_hook(
      d.qc, d.nc, d.qr, d.nr, d.th_atm, d.qv, d.dt, d.qi, d.qm, d.ni,
      d.bm, d.pres, d.dz, d.nc_nuceat_tend, d.nccn_prescribed, d.ni_activated, d.inv_qc_relvar, d.it, d.precip_liq_surf,
      d.precip_ice_surf, d.its, d.ite, d.kts, d.kte, d.diag_eff_radius_qc, d.diag_eff_radius_qi, d.diag_eff_radius_qr,
      d.rho_qi, d.do_predict_nc, d.do_prescribed_CCN, d.use_hetfrz_classnuc, d.dpres, d.inv_exner, d.qv2qi_depos_tend,
      d.precip_liq_flux, d.precip_ice_flux, d.cld_frac_r, d.cld_frac_l, d.cld_frac_i,
      d.liq_ice_exchange, d.vap_liq_exchange, d.vap_ice_exchange, d.qv_prev, d.t_prev, hooks);
    REQUIRE (num_calls == 1);
  }

  // With a sedimentation hook, sedimentation tendencies can be changed by name.
  // Warm columns test liquid (cloud and rain), cold ones ice: in cold columns,
  // rain can freeze within the step, leaving no liquid to sediment.
  void run_sedimentation_hook (const bool warm)
  {
    using P3F   = Functions;
    using SR    = P3SedimentationRates;
    using Hook  = typename P3F::P3SedimentationHook;
    using Hooks = typename P3F::P3Hooks;

    auto engine = Base::get_engine();

    // Columns with cloud and rain, or with ice, which sediment
    //               its, ite, kts, kte, it,  dt, do_predict_nc, do_prescribed_CCN
    P3MainData d_ref(1,   8,   1,  72,  1, 300, true,          false);
    d_ref.randomize(engine, {
        {d_ref.pres           , {5.0e+04, 1.0e+05}},
        {d_ref.dz             , {1.0e+02, 3.0e+02}},
        {d_ref.nc_nuceat_tend , {0      , 0}},
        {d_ref.nccn_prescribed, {0      , 0}},
        {d_ref.ni_activated   , {0      , 0}},
        {d_ref.dpres          , {1.0e+03, 1.5e+03}},
        {d_ref.inv_exner      , {1      , 1}},
        {d_ref.cld_frac_i     , {0.2    , 1}},
        {d_ref.cld_frac_l     , {0.2    , 1}},
        {d_ref.cld_frac_r     , {0.2    , 1}},
        {d_ref.inv_qc_relvar  , {1      , 1}},
        {d_ref.qc             , {1.0e-5 , 1.0e-3}},
        {d_ref.nc             , {1.0e+07, 1.0e+08}},
        {d_ref.qr             , warm ? std::pair<Real,Real>{1.0e-5, 1.0e-3} : std::pair<Real,Real>{0, 0}},
        {d_ref.nr             , warm ? std::pair<Real,Real>{1.0e+03, 1.0e+05} : std::pair<Real,Real>{0, 0}},
        {d_ref.qi             , warm ? std::pair<Real,Real>{0, 0} : std::pair<Real,Real>{1.0e-5, 1.0e-4}},
        {d_ref.qm             , {0      , 0}},
        {d_ref.ni             , warm ? std::pair<Real,Real>{0, 0} : std::pair<Real,Real>{1.0e+04, 1.0e+05}},
        {d_ref.bm             , {0      , 0}},
        {d_ref.qv             , warm ? std::pair<Real,Real>{8.0e-3, 1.2e-2} : std::pair<Real,Real>{1.0e-3, 5.0e-3}},
        {d_ref.qv_prev        , warm ? std::pair<Real,Real>{8.0e-3, 1.2e-2} : std::pair<Real,Real>{1.0e-3, 5.0e-3}},
        {d_ref.th_atm         , warm ? std::pair<Real,Real>{2.85e+02, 2.95e+02} : std::pair<Real,Real>{2.4e+02, 2.6e+02}},
        {d_ref.t_prev         , warm ? std::pair<Real,Real>{2.85e+02, 2.95e+02} : std::pair<Real,Real>{2.4e+02, 2.6e+02}}
    });
    P3MainData d_noop(d_ref), d_reapply(d_ref), d_zero(d_ref), d_wipe(d_ref), d_record(d_ref), d_skip(d_ref);

    auto run = [](P3MainData& d, const Hook& hook) {
      Hooks hooks;
      hooks.sedimentation = hook;
      p3_main_host_hook(
        d.qc, d.nc, d.qr, d.nr, d.th_atm, d.qv, d.dt, d.qi, d.qm, d.ni,
        d.bm, d.pres, d.dz, d.nc_nuceat_tend, d.nccn_prescribed, d.ni_activated, d.inv_qc_relvar, d.it, d.precip_liq_surf,
        d.precip_ice_surf, d.its, d.ite, d.kts, d.kte, d.diag_eff_radius_qc, d.diag_eff_radius_qi, d.diag_eff_radius_qr,
        d.rho_qi, d.do_predict_nc, d.do_prescribed_CCN, d.use_hetfrz_classnuc, d.dpres, d.inv_exner, d.qv2qi_depos_tend,
        d.precip_liq_flux, d.precip_ice_flux, d.cld_frac_r, d.cld_frac_l, d.cld_frac_i,
        d.liq_ice_exchange, d.vap_liq_exchange, d.vap_ice_exchange, d.qv_prev, d.t_prev, hooks);
    };

    const Int ncol = d_ref.ite - d_ref.its + 1, nk = d_ref.kte - d_ref.kts + 1;
    auto hook = [&]() {
      Hook h;
      h.tendencies = typename P3F::template view_3d<Pack>("sed_tend", SR::num_rates, ncol, ekat::npack<Pack>(nk));
      h.before     = typename P3F::template view_3d<Pack>("sed_before", SR::num_rates, ncol, ekat::npack<Pack>(nk));
      return h;
    };
    const int all = (1<<SR::num_rates) - 1;

    run(d_ref, Hook());

    // A hook that changes nothing changes nothing
    int num_calls = 0;
    auto noop = hook();
    noop.callback = [&](const typename P3F::P3SedimentationState& s) {
      ++num_calls;
      REQUIRE (s.ncol == ncol);
      REQUIRE (s.nlev == nk);
      for (const auto name : {"qc", "nc", "qr", "nr", "qi", "ni", "qm", "bm", "rho", "dz", "T_atm"}) {
        REQUIRE (s.state.count(name) == 1);
      }
    };
    run(d_noop, noop);

    // Re-applying the tendencies, and diagnosing the surface precipitation from
    // them, reproduces sedimentation (up to round-off): it conserves mass
    auto reapply = hook();
    reapply.apply_mask = all;
    reapply.diagnose_precip_liq = reapply.diagnose_precip_ice = true;
    reapply.callback = [&](const typename P3F::P3SedimentationState&) { ++num_calls; };
    run(d_reapply, reapply);

    // No sedimentation, by name: no surface precipitation
    auto zero = hook();
    zero.apply_mask = all;
    zero.diagnose_precip_liq = zero.diagnose_precip_ice = true;
    zero.callback = [&](const typename P3F::P3SedimentationState& s) {
      ++num_calls;
      for (int r = 0; r < SR::num_rates; ++r) {
        REQUIRE (SR::index(SR::name(r)) == r);
        Kokkos::deep_copy(Kokkos::subview(s.tendencies, r, Kokkos::ALL, Kokkos::ALL), Pack(0));
      }
    };
    run(d_zero, zero);
    REQUIRE (num_calls == 3);

    if (warm) {
      // Tendencies that remove more than there is: the state is clipped at zero,
      // and the diagnosed precip is what actually left the column
      std::vector<Real> expected_liq(ncol, 0);
      auto wipe = hook();
      wipe.apply_mask = (1<<SR::qc_sed_tend) | (1<<SR::qr_sed_tend);
      wipe.diagnose_precip_liq = true;
      wipe.callback = [&](const typename P3F::P3SedimentationState& s) {
        ++num_calls;
        constexpr Real rho_h2o = 1000;
        auto qc  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.state.at("qc"));
        auto qr  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.state.at("qr"));
        auto rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.state.at("rho"));
        auto dz  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.state.at("dz"));
        for (int i = 0; i < ncol; ++i) {
          for (int k = 0; k < nk; ++k) {
            const int p = k / Pack::n, l = k % Pack::n;
            expected_liq[i] += (qc(i,p)[l] + qr(i,p)[l]) * rho(i,p)[l] * dz(i,p)[l] / (rho_h2o * s.dt);
          }
        }
        for (const int r : {int(SR::qc_sed_tend), int(SR::qr_sed_tend)}) {
          Kokkos::deep_copy(Kokkos::subview(s.tendencies, r, Kokkos::ALL, Kokkos::ALL), Pack(-1.0e3));
        }
      };
      run(d_wipe, wipe);
      REQUIRE (num_calls == 4);
      for (Int i = 0; i < ncol; ++i) {
        REQUIRE (d_wipe.precip_liq_surf[i] == Approx(expected_liq[i]).epsilon(1e-10));
      }

    }

    // Rain (or ice) sedimentation replaced (skipped): its tendencies reach the
    // hook as zero; a hook that puts back P3's own reproduces P3 (to round-off)
    const std::vector<int> species = warm ? std::vector<int>{SR::qr_sed_tend, SR::nr_sed_tend}
      : std::vector<int>{SR::qi_sed_tend, SR::ni_sed_tend, SR::qm_sed_tend, SR::bm_sed_tend};
    using host_tend_t = typename P3F::template view_3d<Pack>::host_mirror_type;
    host_tend_t stock;
    auto record = hook();
    record.callback = [&](const typename P3F::P3SedimentationState& s) {
      stock = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.tendencies);
    };
    run(d_record, record);

    auto skip = hook();
    for (int r : species) skip.skip_mask |= 1 << r;
    skip.apply_mask = skip.skip_mask;
    (warm ? skip.diagnose_precip_liq : skip.diagnose_precip_ice) = true;
    bool rain_was_zero = true;
    skip.callback = [&](const typename P3F::P3SedimentationState& s) {
      auto t = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.tendencies);
      for (const int r : species) {
        for (int i = 0; i < ncol; ++i) {
          for (int k = 0; k < nk; ++k) {
            rain_was_zero &= t(r, i, k / Pack::n)[k % Pack::n] == 0;
          }
          for (int p = 0; p < ekat::npack<Pack>(nk); ++p) {
            t(r, i, p) = stock(r, i, p);
          }
        }
      }
      Kokkos::deep_copy(s.tendencies, t);
    };
    run(d_skip, skip);
    REQUIRE (rain_was_zero);
    // (round-off can move a value across one of P3's thresholds at qsmall)
    constexpr Real qsmall = 1e-14;
    for (Int t = 0; t < d_ref.total(d_ref.qr); ++t) {
      REQUIRE(d_skip.qr[t] == Approx(d_ref.qr[t]).epsilon(1e-10).margin(qsmall));
      REQUIRE(d_skip.qi[t] == Approx(d_ref.qi[t]).epsilon(1e-10).margin(qsmall));
      REQUIRE(d_skip.qc[t] == Approx(d_ref.qc[t]).epsilon(1e-10).margin(qsmall));
    }
    for (Int i = 0; i < ncol; ++i) {
      REQUIRE(d_skip.precip_liq_surf[i] == Approx(d_ref.precip_liq_surf[i]).epsilon(1e-8).margin(1e-20));
      REQUIRE(d_skip.precip_ice_surf[i] == Approx(d_ref.precip_ice_surf[i]).epsilon(1e-8).margin(1e-20));
    }

    const auto tot = d_ref.total(d_ref.qc);
    for (Int t = 0; t < tot; ++t) {
      REQUIRE(d_noop.qc[t] == d_ref.qc[t]);
      REQUIRE(d_noop.qr[t] == d_ref.qr[t]);
      REQUIRE(d_noop.nr[t] == d_ref.nr[t]);
      REQUIRE(d_noop.qi[t] == d_ref.qi[t]);
      REQUIRE(d_noop.ni[t] == d_ref.ni[t]);
      REQUIRE(d_noop.th_atm[t] == d_ref.th_atm[t]);
      REQUIRE(d_reapply.qc[t] == Approx(d_ref.qc[t]).epsilon(1e-10).margin(qsmall));
      REQUIRE(d_reapply.qr[t] == Approx(d_ref.qr[t]).epsilon(1e-10).margin(qsmall));
      REQUIRE(d_reapply.qi[t] == Approx(d_ref.qi[t]).epsilon(1e-10).margin(qsmall));
    }
    Real liq_max = 0, ice_max = 0;
    for (Int i = 0; i < ncol; ++i) {
      REQUIRE(d_noop.precip_liq_surf[i] == d_ref.precip_liq_surf[i]);
      REQUIRE(d_noop.precip_ice_surf[i] == d_ref.precip_ice_surf[i]);
      REQUIRE(d_reapply.precip_liq_surf[i] == Approx(d_ref.precip_liq_surf[i]).epsilon(1e-8).margin(1e-20));
      REQUIRE(d_reapply.precip_ice_surf[i] == Approx(d_ref.precip_ice_surf[i]).epsilon(1e-8).margin(1e-20));
      REQUIRE(d_zero.precip_liq_surf[i] == 0);
      REQUIRE(d_zero.precip_ice_surf[i] == 0);
      liq_max = std::max(liq_max, d_ref.precip_liq_surf[i]);
      ice_max = std::max(ice_max, d_ref.precip_ice_surf[i]);
    }
    REQUIRE ((warm ? liq_max : ice_max) > 0);
  }
#endif
};

} // namespace unit_test
} // namespace p3
} // namespace scream

namespace {

TEST_CASE("p3_process_rates_registry", "[p3_functions]")
{
  using T = scream::p3::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestP3ProcessRates;

  T t; t.run_registry();
}

#ifdef SCREAM_P3_SMALL_KERNELS
TEST_CASE("p3_process_rates_hook", "[p3_functions]")
{
  using T = scream::p3::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestP3ProcessRates;

  T t; t.run_hook();
}

TEST_CASE("p3_process_rates_quiet_columns", "[p3_functions]")
{
  using T = scream::p3::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestP3ProcessRates;

  T t; t.run_quiet_columns();
}

TEST_CASE("p3_sedimentation_hook", "[p3_functions]")
{
  using T = scream::p3::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestP3ProcessRates;

  T t;
  SECTION ("liquid") { t.run_sedimentation_hook(true); }
  SECTION ("ice")    { t.run_sedimentation_hook(false); }
}
#endif

} // namespace
