#include "catch2/catch.hpp"

#include "p3_functions.hpp"
#include "p3_test_data.hpp"
#include "p3_unit_tests_common.hpp"
#include "warm_rain_emulator/p3_warm_rain_mlp.hpp"

#include "share/core/eamxx_types.hpp"

#include <array>
#include <cmath>
#include <fstream>

namespace scream {
namespace p3 {
namespace unit_test {

/*
 * Property tests of the warm-rain emulator merge (warm_rain_emulator_merge).
 * The merge converts grid-mean emulated rates to the in-cloud rates that part2
 * expects, so these tests map the merged rates back to cell averages with the
 * same cloud fractions as back_to_cell_average and check them.
 */
template <typename D>
struct UnitWrap::UnitTest<D>::TestWarmRainEmulatorMerge : public UnitWrap::UnitTest<D>::Base {

  // Order of P3WarmRainRates
  static constexpr int nrates = 7;
  enum { AUTOCONV, NC_AUTOCONV, NCAUTR, NC_SELFCOLL, ACCRET, NC_ACCRET, NR_SELFCOLL };
  using Rates = std::array<Scalar, nrates>;

  struct Case {
    Scalar cld_frac_l, cld_frac_r, nc, nr, dt, kk_factor;
    bool do_cloud_self_collection;
    Rates emu;     // grid mean
    Scalar use_cloud, use_rain;
    Rates stock;   // in cloud
    bool context;
  };

  // Run the merge on device for one case, and return the merged in-cloud rates
  static Rates merge (const Case& c)
  {
    view_1d<Scalar> out("out", nrates);
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(const Int&) {
      Pack r[nrates];
      for (int n = 0; n < nrates; ++n) { r[n] = c.stock[n]; }
      Functions::warm_rain_emulator_merge(
        c.cld_frac_l, c.cld_frac_r, c.nc, c.nr, 1/c.dt, c.kk_factor, c.do_cloud_self_collection,
        c.emu[0], c.emu[1], c.emu[2], c.emu[3], c.emu[4], c.emu[5], c.emu[6],
        c.use_cloud, c.use_rain,
        r[0], r[1], r[2], r[3], r[4], r[5], r[6], Mask(c.context));
      for (int n = 0; n < nrates; ++n) { out(n) = r[n][0]; }
    });
    const auto out_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    Rates res;
    for (int n = 0; n < nrates; ++n) { res[n] = out_h(n); }
    return res;
  }

  // Cloud fractions back_to_cell_average multiplies each warm-rain rate by
  static Rates cell_fractions (const Scalar cld_frac_l, const Scalar cld_frac_r)
  {
    const Scalar lr = std::min(cld_frac_l, cld_frac_r);
    return {cld_frac_l, cld_frac_l, lr, cld_frac_l, lr, lr, cld_frac_r};
  }

  static Case base_case ()
  {
    Case c;
    c.cld_frac_l = 0.3; c.cld_frac_r = 0.5;
    c.nc = 3e7; c.nr = 1.8e3; c.dt = 100; c.kk_factor = 0.36;
    c.do_cloud_self_collection = true;
    c.emu   = {4.7e-10, 3.5, 1.8, -446.3, 5.5e-9, 739.9, 0.65};
    c.stock = {1e-9, 7.0, 3.0, 0, 1e-8, 900.0, 2.0};
    c.use_cloud = 1; c.use_rain = 1;
    c.context = true;
    return c;
  }

  void run_phys ()
  {
    const Scalar tol = std::numeric_limits<Scalar>::epsilon()*10;

    // 1. Emulated rates enter part2 unchanged: no rescaling by cloud fractions
    {
      const auto c = base_case();
      const auto r = merge(c);
      const auto f = cell_fractions(c.cld_frac_l, c.cld_frac_r);
      for (int n = 0; n < nrates; ++n) {
        REQUIRE (r[n]*f[n] == Approx(c.emu[n]).epsilon(tol));
      }
    }

    // 2. Outside the envelope: stock P3, with the three autoconversion rates scaled
    {
      auto c = base_case();
      c.use_cloud = 0; c.use_rain = 0;
      const auto r = merge(c);
      for (int n : {AUTOCONV, NC_AUTOCONV, NCAUTR}) {
        REQUIRE (r[n] == Approx(c.stock[n]*c.kk_factor).epsilon(tol));
      }
      for (int n : {NC_SELFCOLL, ACCRET, NC_ACCRET, NR_SELFCOLL}) {
        REQUIRE (r[n] == c.stock[n]);
      }
    }

    // 3. Cloud and rain masks are independent
    {
      auto c = base_case();
      c.use_cloud = 1; c.use_rain = 0;
      auto r = merge(c);
      REQUIRE (r[NR_SELFCOLL] == c.stock[NR_SELFCOLL]);
      REQUIRE (r[ACCRET]*c.cld_frac_l == Approx(c.emu[ACCRET]).epsilon(tol));

      c.use_cloud = 0; c.use_rain = 1;
      r = merge(c);
      REQUIRE (r[NR_SELFCOLL]*c.cld_frac_r == Approx(c.emu[NR_SELFCOLL]).epsilon(tol));
      REQUIRE (r[ACCRET] == c.stock[ACCRET]);
    }

    // 4. Cloud self-collection can be turned off
    {
      auto c = base_case();
      c.do_cloud_self_collection = false;
      const auto r = merge(c);
      REQUIRE (r[NC_SELFCOLL] == 0);
    }

    // 5. Caps: with a huge dt, self-collection cannot remove more than what is there
    {
      auto c = base_case();
      c.dt = 1e6;
      c.emu[NC_SELFCOLL] = -1e10;
      c.emu[NR_SELFCOLL] = 1e10;
      const auto r = merge(c);
      const auto f = cell_fractions(c.cld_frac_l, c.cld_frac_r);
      REQUIRE (r[NC_SELFCOLL]*f[NC_SELFCOLL] == Approx(-c.nc/c.dt).epsilon(tol));
      REQUIRE (r[NR_SELFCOLL]*f[NR_SELFCOLL] == Approx(c.nr/c.dt + c.emu[NCAUTR]).epsilon(tol));
      REQUIRE (c.nc + r[NC_SELFCOLL]*f[NC_SELFCOLL]*c.dt >= 0);
    }

    // 6. Nothing changes outside of the context (e.g., pack padding)
    {
      auto c = base_case();
      c.context = false;
      const auto r = merge(c);
      for (int n = 0; n < nrates; ++n) {
        REQUIRE (r[n] == c.stock[n]);
      }
    }
  }

  // Device-native MLP backend: a synthetic one-layer model whose outputs are known
  void run_mlp ()
  {
    using MLP = WarmRainMLP<Pack, D>;

    // Zero weights, so the network output is softplus(bias)
    const std::string fname = "p3_warm_rain_mlp_test.txt";
    {
      std::ofstream f(fname);
      f << "p3_warm_rain_mlp 1\n"
        << "activation tanh softplus\n"
        << "widths 1 4 4\n"
        << "x_mean 0 0 0 0\nx_std 1 1 1 1\nfloors 1e-12 1 1e-12 1e-3\n"
        << "y_scale 1e-9 1e-8 400 0.2\ny_log_std 1 1 0.5 1.5\n"
        << "gates 1e-5 1e-7\n"
        << "envelope_cloud 1e-3 3e5 1e8 1e-7 1e-4 1e5\n"
        << "envelope_rain 1e-4 1e5\n"
        << "number_rates 4e-5 2 0.9\n"
        << "weight 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        << "bias 0.1 0.2 0.3 0.4\n"
        << "end\n";
    }
    const auto mlp = MLP::load(fname);

    const Scalar y_scale[4] = {1e-9, 1e-8, 400, 0.2}, y_log_std[4] = {1, 1, 0.5, 1.5}, bias[4] = {0.1, 0.2, 0.3, 0.4};
    Scalar raw[4];
    for (int o = 0; o < 4; ++o) {
      raw[o] = std::expm1(std::log1p(std::exp(bias[o]))*y_log_std[o])*y_scale[o];
    }

    // Cases: per-kg qc, nc, qr, nr, and rho. Expected masks: in envelope (cloud), and rain
    struct In { Scalar qc, nc, qr, nr, rho, use_cloud, use_rain; };
    const In cases[] = {
      {2e-4, 3e7, 3e-6, 1.8e3, 1.0, 1, 1},  // inside the envelope
      {2e-4, 3e7, 3e-6, 1.8e3, 1.1, 1, 1},  // same, different density
      {5e-6, 3e7, 3e-6, 1.8e3, 1.0, 0, 1},  // below the qc gate: no cloud processes
      {2e-4, 3e7, 0,    0,     1.0, 0, 1},  // no rain: no accretion, no rain self-collection
      {3e-3, 3e7, 3e-6, 1.8e3, 1.0, 0, 1},  // too much cloud water for the envelope
    };
    for (const auto& c : cases) {
      view_1d<Scalar> out("out", 9);
      Kokkos::parallel_for(1, KOKKOS_LAMBDA(const Int&) {
        Pack r[9];
        mlp.p3_rates(c.qc, c.nc, c.qr, c.nr, c.rho, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
        for (int n = 0; n < 9; ++n) { out(n) = r[n][0]; }
      });
      const auto o = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);

      // Gates, applied to per-volume values
      const bool cloud = c.qc*c.rho > 1e-5, rain = c.qr*c.rho > 1e-7;
      const Scalar au  = cloud ? raw[0]/c.rho : 0;
      const Scalar ac  = cloud && c.qr > 0 ? raw[1]/c.rho : 0;
      const Scalar scc = cloud ? raw[2]/c.rho : 0;
      const Scalar scr = rain  ? raw[3]/c.rho : 0;
      const Scalar m_star = 4.0/3.0*M_PI*1000*std::pow(4e-5, 3);
      const Scalar tol = 1e-12;
      REQUIRE (o(0) == Approx(au).epsilon(tol));                      // qc2qr_autoconv_tend
      REQUIRE (o(1) == Approx(2*au/m_star).epsilon(tol));             // nc2nr_autoconv_tend
      REQUIRE (o(2) == Approx(au/m_star).epsilon(tol));               // ncautr
      REQUIRE (o(3) == Approx(-scc).epsilon(tol));                    // nc_selfcollect_tend
      REQUIRE (o(4) == Approx(ac).epsilon(tol));                      // qc2qr_accret_tend
      REQUIRE (o(5) == Approx(0.9*ac*c.nc/c.qc).epsilon(tol));        // nc_accret_tend
      REQUIRE (o(6) == Approx(scr).epsilon(tol));                     // nr_selfcollect_tend
      REQUIRE (o(7) == c.use_cloud);
      REQUIRE (o(8) == c.use_rain);
    }
    std::remove(fname.c_str());
  }

#ifdef SCREAM_P3_SMALL_KERNELS
  // The warm-rain hook runs between the warm-rain stage and part2, and part2 uses
  // whatever rates it leaves.
  void run_hook ()
  {
    using P3F = Functions;

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

    auto run = [](P3MainData& d, const typename P3F::WarmRainHook& hook) {
      p3_main_host_hook(
        d.qc, d.nc, d.qr, d.nr, d.th_atm, d.qv, d.dt, d.qi, d.qm, d.ni,
        d.bm, d.pres, d.dz, d.nc_nuceat_tend, d.nccn_prescribed, d.ni_activated, d.inv_qc_relvar, d.it, d.precip_liq_surf,
        d.precip_ice_surf, d.its, d.ite, d.kts, d.kte, d.diag_eff_radius_qc, d.diag_eff_radius_qi, d.diag_eff_radius_qr,
        d.rho_qi, d.do_predict_nc, d.do_prescribed_CCN, d.use_hetfrz_classnuc, d.dpres, d.inv_exner, d.qv2qi_depos_tend,
        d.precip_liq_flux, d.precip_ice_flux, d.cld_frac_r, d.cld_frac_l, d.cld_frac_i,
        d.liq_ice_exchange, d.vap_liq_exchange, d.vap_ice_exchange, d.qv_prev, d.t_prev, hook);
    };

    int num_calls = 0;
    run(d_ref,  typename P3F::WarmRainHook());
    run(d_noop, [&](const typename P3F::P3Temporaries&) { ++num_calls; });
    run(d_zero, [&](const typename P3F::P3Temporaries& t) {
      ++num_calls;
      Kokkos::deep_copy(t.warm_rain.qc2qr_autoconv_tend, 0);
      Kokkos::deep_copy(t.warm_rain.nc2nr_autoconv_tend, 0);
      Kokkos::deep_copy(t.warm_rain.ncautr, 0);
      Kokkos::deep_copy(t.warm_rain.nc_selfcollect_tend, 0);
      Kokkos::deep_copy(t.warm_rain.qc2qr_accret_tend, 0);
      Kokkos::deep_copy(t.warm_rain.nc_accret_tend, 0);
      Kokkos::deep_copy(t.warm_rain.nr_selfcollect_tend, 0);
    });
    REQUIRE (num_calls == 2);

    // A hook that changes nothing changes nothing
    const auto tot = d_ref.total(d_ref.qc);
    for (Int t = 0; t < tot; ++t) {
      REQUIRE(d_noop.qc[t] == d_ref.qc[t]);
      REQUIRE(d_noop.nc[t] == d_ref.nc[t]);
      REQUIRE(d_noop.qr[t] == d_ref.qr[t]);
      REQUIRE(d_noop.nr[t] == d_ref.nr[t]);
      REQUIRE(d_noop.qv[t] == d_ref.qv[t]);
      REQUIRE(d_noop.th_atm[t] == d_ref.th_atm[t]);
    }

    // Without warm-rain processes, no rain forms
    Real qr_max_ref = 0;
    for (Int t = 0; t < tot; ++t) {
      qr_max_ref = std::max(qr_max_ref, d_ref.qr[t]);
      REQUIRE(d_zero.qr[t] == 0);
    }
    REQUIRE (qr_max_ref > 0);
  }
#endif
};

} // namespace unit_test
} // namespace p3
} // namespace scream

namespace {

TEST_CASE("p3_warm_rain_emulator_merge", "[p3_functions]")
{
  using T = scream::p3::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestWarmRainEmulatorMerge;

  T t; t.run_phys();
}

TEST_CASE("p3_warm_rain_mlp", "[p3_functions]")
{
  using T = scream::p3::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestWarmRainEmulatorMerge;

  T t; t.run_mlp();
}

#ifdef SCREAM_P3_SMALL_KERNELS
TEST_CASE("p3_warm_rain_hook", "[p3_functions]")
{
  using T = scream::p3::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestWarmRainEmulatorMerge;

  T t; t.run_hook();
}
#endif

} // namespace
