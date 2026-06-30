#include <catch2/catch.hpp>

#include "share/stochastic/spectral_pattern_generator.hpp"
#include "share/grid/point_grid.hpp"
#include "share/field/field.hpp"
#include "share/field/field_identifier.hpp"
#include "share/field/field_layout.hpp"

#include <ekat_comm.hpp>
#include <ekat_units.hpp>

#include <cmath>
#include <vector>

namespace {

using namespace scream;
using namespace scream::stochastic;

SpectralPatternGenerator::Params make_params (int seed=42, int member=0)
{
  SpectralPatternGenerator::Params p;
  p.truncation  = 12;
  p.nmin        = 1;
  p.power       = 1.5;
  p.stddev      = 0.5;
  p.decorr_time = 4.0*3600.0;
  p.base_seed   = seed;
  p.member_id   = member;
  return p;
}

std::vector<Real> coeffs_host (const SpectralPatternGenerator& g)
{
  Field cf = g.get_coefficient_field();           // shallow handle, shares data
  cf.sync_to_host();                               // coefficients are device-authoritative
  auto h = cf.get_view<const Real*,Host>();
  std::vector<Real> out(h.extent_int(0));
  for (int i=0; i<h.extent_int(0); ++i) out[i] = h(i);
  return out;
}

} // anonymous namespace

TEST_CASE ("spectral_pattern_generator")
{
  ekat::Comm comm(MPI_COMM_WORLD);
  const int ncols = 32, nlevs = 4;
  auto grid = create_point_grid("Physics", ncols, nlevs, comm);
  const double dt = 1800.0;

  SECTION ("determinism_same_seed") {
    SpectralPatternGenerator a, b;
    a.initialize("a", grid, make_params(7), comm);
    b.initialize("b", grid, make_params(7), comm);
    for (int s=0; s<20; ++s) { a.advance(dt); b.advance(dt); }
    auto ca = coeffs_host(a), cb = coeffs_host(b);
    REQUIRE(ca.size()==cb.size());
    for (size_t i=0;i<ca.size();++i) REQUIRE(ca[i]==cb[i]);   // bit-for-bit
  }

  SECTION ("different_members_differ") {
    SpectralPatternGenerator a, b;
    a.initialize("a", grid, make_params(7,0), comm);
    b.initialize("b", grid, make_params(7,1), comm);
    for (int s=0; s<20; ++s) { a.advance(dt); b.advance(dt); }
    auto ca = coeffs_host(a), cb = coeffs_host(b);
    Real maxdiff = 0;
    for (size_t i=0;i<ca.size();++i) maxdiff = std::max(maxdiff, std::abs(ca[i]-cb[i]));
    REQUIRE(maxdiff > 0);
  }

  SECTION ("restart_reproducibility") {
    // Advance K steps, snapshot (coeffs,step); advance M more for the reference.
    SpectralPatternGenerator a;
    a.initialize("a", grid, make_params(11), comm);
    const int K=15, M=10;
    for (int s=0;s<K;++s) a.advance(dt);
    auto coeffs_K = coeffs_host(a);
    const auto step_K = a.step_counter();
    for (int s=0;s<M;++s) a.advance(dt);
    auto coeffs_ref = coeffs_host(a);

    // Fresh generator "restarted" from the K-step state, advanced M more.
    SpectralPatternGenerator c;
    c.initialize("c", grid, make_params(11), comm);
    {
      Field cf = c.get_coefficient_field();
      auto h = cf.get_view<Real*,Host>();
      for (int i=0;i<h.extent_int(0);++i) h(i) = coeffs_K[i];
      cf.sync_to_dev();                            // make the device state authoritative
      c.set_step_counter(step_K);
    }
    for (int s=0;s<M;++s) c.advance(dt);
    auto coeffs_c = coeffs_host(c);
    for (size_t i=0;i<coeffs_ref.size();++i) REQUIRE(coeffs_ref[i]==coeffs_c[i]);
  }

  SECTION ("ar1_autocorrelation") {
    auto p = make_params(3);
    SpectralPatternGenerator g;
    g.initialize("g", grid, p, comm);
    const Real alpha = std::exp(-dt/p.decorr_time);
    const int idx = sph_idx_cos(2,1,p.truncation);

    // Spin up, then record a coefficient time series.
    for (int s=0;s<200;++s) g.advance(dt);
    std::vector<Real> x;
    for (int s=0;s<4000;++s) { g.advance(dt); x.push_back(coeffs_host(g)[idx]); }

    double mean=0; for (auto v:x) mean+=v; mean/=x.size();
    double num=0, den=0;
    for (size_t t=1;t<x.size();++t) { num += (x[t]-mean)*(x[t-1]-mean); }
    for (auto v:x) den += (v-mean)*(v-mean);
    const double rho = num/den;
    REQUIRE(std::abs(rho - alpha) < 0.1);
  }

  SECTION ("evaluate_pattern_zero_mean_ish") {
    using namespace ShortFieldTagsNames;
    SpectralPatternGenerator g;
    g.initialize("g", grid, make_params(5), comm);

    // Build lat/lon fields over a dense, near-uniform set of points.
    const int npts = 2000;
    FieldLayout fl({COL},{npts});
    FieldIdentifier lat_fid("lat", fl, ekat::units::none, grid->name());
    FieldIdentifier lon_fid("lon", fl, ekat::units::none, grid->name());
    Field lat(lat_fid), lon(lon_fid);
    lat.allocate_view(); lon.allocate_view();
    auto lath = lat.get_view<Real*,Host>();
    auto lonh = lon.get_view<Real*,Host>();
    // Roughly area-uniform: equal-area in sin(lat), uniform in lon.
    for (int i=0;i<npts;++i) {
      const Real u = (i + 0.5)/npts;          // in (0,1)
      lath(i) = std::asin(2*u-1) * 180.0/M_PI; // deg, equal-area
      lonh(i) = std::fmod(i*2.399963, 2*M_PI) * 180.0/M_PI; // golden-angle spread, deg
    }
    lat.sync_to_dev(); lon.sync_to_dev();
    g.set_columns(lat, lon);

    for (int s=0;s<50;++s) g.advance(dt);

    SpectralPatternGenerator::view_1d<Real> pat("pat", npts);
    g.evaluate_pattern(pat);
    auto path = Kokkos::create_mirror_view(pat);
    Kokkos::deep_copy(path, pat);

    double mean=0, var=0;
    for (int i=0;i<npts;++i) mean += path(i);
    mean /= npts;
    for (int i=0;i<npts;++i) var += (path(i)-mean)*(path(i)-mean);
    var /= npts;
    const double stddev = std::sqrt(var);
    REQUIRE(stddev > 0);
    // Sample mean should be small relative to the spread (modes have zero
    // spherical mean; finite sampling leaves a small residual).
    REQUIRE(std::abs(mean) < 0.5*stddev);
  }
}
