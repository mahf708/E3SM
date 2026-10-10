#include "catch2/catch.hpp"

#include "share/core/eamxx_types.hpp"
#include "shoc_functions.hpp"
#include "shoc_test_data.hpp"

#include "shoc_unit_tests_common.hpp"

namespace scream {
namespace shoc {
namespace unit_test {

template <typename D>
struct UnitWrap::UnitTest<D>::TestShocHooks : public UnitWrap::UnitTest<D>::Base {

#ifdef SCREAM_SHOC_SMALL_KERNELS
  // With an eddy-diffusivity hook, the implicit diffusion solver uses the
  // eddy diffusivities the hook leaves
  void run_eddy_diffusivities ()
  {
    using SHF   = scream::shoc::Functions<Real, DefaultDevice>;
    using Hooks = typename SHF::SHOCHooks;
    using State = typename SHF::SHOCEddyDiffusivityState;

    auto engine = Base::get_engine();

    //               shcol, nlev, nlevi, num_qtracers, dtime, nadv, nbot_shoc, ntop_shoc
    ShocMainData d_ref(7,     16,    17,            3,   300,    2,        12, 0);
    d_ref.randomize(engine, {
        {d_ref.presi, {700e2,1000e2}},
        {d_ref.tkh, {3,50}},
        {d_ref.tke, {0.1,0.3}},
        {d_ref.zi_grid, {0, 3000}},
        {d_ref.wthl_sfc, {0,1e-4}},
        {d_ref.wqw_sfc, {0,1e-6}},
        {d_ref.uw_sfc, {0,1e-2}},
        {d_ref.vw_sfc, {0,1e-4}},
        {d_ref.host_dx, {3000, 3000}},
        {d_ref.host_dy, {3000, 3000}},
        {d_ref.phis, {0, 500}},
        {d_ref.wthv_sec, {-0.02, 0.03}},
        {d_ref.qw, {1e-4, 5e-2}},
        {d_ref.u_wind, {-10, 0}},
        {d_ref.v_wind, {-10, 0}},
        {d_ref.shoc_ql, {0, 1e-3}},
      });
    ShocMainData d_init(d_ref), d_noop(d_ref), d_zero(d_ref);
    const Int shcol = d_ref.shcol, nlev = d_ref.nlev;

    shoc_main(d_ref);

    // A hook that changes nothing changes nothing; it runs in each of the nadv steps
    int num_calls = 0;
    Hooks noop;
    noop.eddy_diffusivities = [&](const State& s) {
      ++num_calls;
      REQUIRE (s.ncol == shcol);
      REQUIRE (s.nlev == nlev);
      REQUIRE (s.nlevi == nlev+1);
      for (const auto name : {"tk", "tkh", "tke", "isotropy"}) {
        REQUIRE (s.outputs.count(name) == 1);
      }
      REQUIRE (s.state.count("shoc_mix") == 1);
      REQUIRE (s.interface_state.count("zi_grid") == 1);
      REQUIRE (s.column_state.count("pblh") == 1);
    };
    shoc_main(d_noop, noop);
    REQUIRE (num_calls == d_ref.nadv);

    // No eddy diffusivities: winds only change at the bottom, through the surface fluxes
    Hooks zero;
    zero.eddy_diffusivities = [&](const State& s) {
      Kokkos::deep_copy(s.outputs.at("tk"), 0);
      Kokkos::deep_copy(s.outputs.at("tkh"), 0);
    };
    shoc_main(d_zero, zero);

    bool differs = false;
    for (Int i = 0; i < shcol; ++i) {
      for (Int k = 0; k < nlev; ++k) {
        const auto o = k + i*nlev;
        REQUIRE (d_noop.thetal[o] == d_ref.thetal[o]);
        REQUIRE (d_noop.qw[o]     == d_ref.qw[o]);
        REQUIRE (d_noop.tke[o]    == d_ref.tke[o]);
        REQUIRE (d_noop.u_wind[o] == d_ref.u_wind[o]);
        REQUIRE (d_noop.tk[o]     == d_ref.tk[o]);
        if (k < nlev-1) {
          REQUIRE (d_zero.u_wind[o] == d_init.u_wind[o]);
          REQUIRE (d_zero.v_wind[o] == d_init.v_wind[o]);
        }
        differs |= d_ref.u_wind[o] != d_init.u_wind[o];
      }
    }
    REQUIRE (differs);
  }
#endif
};

} // namespace unit_test
} // namespace shoc
} // namespace scream

namespace {

#ifdef SCREAM_SHOC_SMALL_KERNELS
TEST_CASE("shoc_eddy_diffusivities_hook", "[shoc_functions]")
{
  using T = scream::shoc::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestShocHooks;

  T t; t.run_eddy_diffusivities();
}
#endif

} // namespace
