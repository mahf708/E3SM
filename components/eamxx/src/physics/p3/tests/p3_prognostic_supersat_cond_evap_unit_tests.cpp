#include "catch2/catch.hpp"

#include "p3_functions.hpp"
#include "p3_test_data.hpp"
#include "p3_unit_tests_common.hpp"

#include "share/physics/physics_constants.hpp"
#include "share/core/eamxx_types.hpp"

#include <cmath>

namespace scream {
namespace p3 {
namespace unit_test {

/*
 * Property tests for the prognostic supersaturation condensation/evaporation
 * rate used when the p3_super_sat runtime option is enabled. This is the
 * routine that replaces the saturation adjustment normally done by the
 * macrophysics: instead of forcing the grid box back to saturation, cloud
 * liquid is condensed/evaporated at a finite rate set by the local
 * supersaturation and the supersaturation relaxation timescale.
 */
template <typename D>
struct UnitWrap::UnitTest<D>::TestPrognosticSupersatCondEvap : public UnitWrap::UnitTest<D>::Base {

  // A physically reasonable base state, roughly 800 hPa in a shallow cumulus
  struct BaseState {
    Pack qc_incld = 1.0e-4;  // in-cloud liquid                    [kg/kg]
    Pack qv_sat_l = 5.0e-3;  // saturation vapor mixing ratio      [kg/kg]
    Pack qv       = 5.0e-3;  // vapor mixing ratio                 [kg/kg]
    Pack dqsdt    = 3.0e-4;  // d(qv_sat_l)/dT                     [kg/kg/K]
    Pack ab       = 2.0;     // latent heating correction factor   [-]
    Pack epsc     = 5.0e-2;  // inverse relaxation timescale       [1/s]
    Pack rho      = 1.0;     // air density                        [kg/m3]
    Pack omega_mp = 0;       // vertical pressure velocity         [Pa/s]
    Pack tke_mp   = 0;       // turbulent kinetic energy           [m2/s2]
    Scalar dt     = 60;      // time step                          [s]
  };

  static void call(const BaseState& s, Pack& w_updraft, Pack& tend,
                   const Mask& context = Mask(true))
  {
    Functions::prognostic_supersat_cond_evap(s.qc_incld, s.qv, s.qv_sat_l, s.dqsdt, s.ab,
                                             s.epsc, s.rho, s.omega_mp, s.tke_mp,
                                             s.dt, 1/s.dt, w_updraft, tend, context);
  }

  void run_property()
  {
    constexpr Scalar gravit = C::gravit.value;

    // TEST ONE
    // Saturated air with no resolved or sub-grid vertical motion has neither a
    // supersaturation nor a source of one, so there must be no condensation or
    // evaporation at all.
    {
      BaseState s;
      s.qv = s.qv_sat_l;

      Pack w_updraft(0), tend(0);
      call(s, w_updraft, tend);

      REQUIRE(w_updraft[0] == 0);
      REQUIRE(tend[0] == 0);
    }

    // TEST TWO
    // Supersaturated air must condense (positive tendency), and subsaturated
    // air must evaporate (negative tendency).
    {
      BaseState s;
      Pack w_updraft(0), tend_super(0), tend_sub(0);

      s.qv = s.qv_sat_l * 1.01;
      call(s, w_updraft, tend_super);
      REQUIRE(tend_super[0] > 0);

      s.qv = s.qv_sat_l * 0.99;
      call(s, w_updraft, tend_sub);
      REQUIRE(tend_sub[0] < 0);
    }

    // TEST THREE
    // Evaporation is limited so that it can never remove more liquid than is
    // actually present over the time step. Drive it with completely dry air,
    // which without the limiter would ask for a far larger sink.
    {
      BaseState s;
      s.qv = 0;

      Pack w_updraft(0), tend(0);
      call(s, w_updraft, tend);

      REQUIRE(tend[0] < 0);
      REQUIRE(tend[0] == -s.qc_incld[0] / s.dt);
    }

    // TEST FOUR
    // The updraft speed is the resolved part from omega plus a sub-grid part
    // from TKE, assuming isotropic turbulence so that w'^2 = (2/3) tke.
    {
      BaseState s;
      s.omega_mp = -1;   // ascent
      s.tke_mp   = 0.3;

      Pack w_updraft(0), tend(0);
      call(s, w_updraft, tend);

      // Loose relative tolerance so that this holds in single precision too
      const Scalar w_expected = 1 / gravit + std::sqrt(2 * C::THIRD * Scalar(0.3));
      REQUIRE(std::abs(w_updraft[0] - w_expected) < 1e-5 * w_expected);
      REQUIRE(w_updraft[0] > 0);
    }

    // TEST FIVE
    // Starting from exact saturation, ascent supplies supersaturation and must
    // produce condensation, and a stronger ascent must produce more of it.
    // Subsidence must instead produce evaporation.
    {
      BaseState s;
      s.qv = s.qv_sat_l;

      Pack w_updraft(0), tend_weak(0), tend_strong(0), tend_subsidence(0);

      s.omega_mp = -1;
      call(s, w_updraft, tend_weak);
      REQUIRE(tend_weak[0] > 0);

      s.omega_mp = -5;
      call(s, w_updraft, tend_strong);
      REQUIRE(tend_strong[0] > tend_weak[0]);

      // Subsidence with no sub-grid contribution warms the parcel and evaporates
      s.omega_mp = 1;
      s.tke_mp   = 0;
      call(s, w_updraft, tend_subsidence);
      REQUIRE(w_updraft[0] < 0);
      REQUIRE(tend_subsidence[0] < 0);
    }

    // TEST SIX
    // Regression test for the masked lanes. epsc and ab are only meaningful
    // where the context mask is set; the routine must not divide by them
    // elsewhere. Every lane must come back finite, and the inactive lanes must
    // be left at exactly zero.
    if (Pack::n > 1) {
      BaseState s;
      s.qv = s.qv_sat_l * 1.01;

      // Mimic what p3_main_part2 passes in: zeros outside the active lanes
      s.epsc = 0;
      s.ab   = 0;
      s.epsc[0] = 5.0e-2;
      s.ab[0]   = 2.0;

      Mask context(false);
      context.set(0, true);

      Pack w_updraft(0), tend(0);
      call(s, w_updraft, tend, context);

      REQUIRE(tend[0] > 0);
      for (Int i = 0; i < Pack::n; ++i) {
        REQUIRE(std::isfinite(w_updraft[i]));
        REQUIRE(std::isfinite(tend[i]));
      }
      for (Int i = 1; i < Pack::n; ++i) {
        REQUIRE(w_updraft[i] == 0);
        REQUIRE(tend[i] == 0);
      }
    }

    // TEST SEVEN
    // An empty context must leave both outputs untouched at zero.
    {
      BaseState s;
      s.qv = s.qv_sat_l * 1.01;

      Pack w_updraft(0), tend(0);
      call(s, w_updraft, tend, Mask(false));

      for (Int i = 0; i < Pack::n; ++i) {
        REQUIRE(w_updraft[i] == 0);
        REQUIRE(tend[i] == 0);
      }
    }
  }

};

} // namespace unit_test
} // namespace p3
} // namespace scream

namespace {

TEST_CASE("prognostic_supersat_cond_evap_property", "[p3]")
{
  using T = scream::p3::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestPrognosticSupersatCondEvap;

  T t;
  t.run_property();
}

} // empty namespace
