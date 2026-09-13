/**
 * @file sea_ice_surface.cpp
 * @brief The sea ice surface.  The bulk formulae are dice's, by way of
 *        EICE's eice_flux_atmice_mod.F90 and ice_comp_mct.F90 on
 *        mahf708/eocn/add-samudra; the tests check them against that
 *        Fortran on the same inputs.
 */

#include "sea_ice_surface.hpp"

#include "calendar.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace emulator {
namespace ice {

namespace {

constexpr double umin = 1.0;      // m/s, minimum wind speed
constexpr double zref = 10.0;     // m, reference height
constexpr double ztref = 2.0;     // m, reference height for air temperature
constexpr double zzsice = 0.0005; // m, ice surface roughness

/// Saturation humidity of air, kg/m3.
double qsat(double tk) { return 627572.4 / std::exp(5107.4 / tk); }

double psimhu(double x) {
  return std::log((1.0 + x * (2.0 + x)) * (1.0 + x * x) / 8.0) -
         2.0 * std::atan(x) + 1.571;
}

double psixhu(double x) { return 2.0 * std::log((1.0 + x * x) / 2.0); }

} // namespace

bool bulk_fluxes_defined(const AtmosphereAtIce &atm) {
  return atm.dens > 0.0 && atm.z > 0.0 && std::isfinite(atm.dens) &&
         std::isfinite(atm.z);
}

AtmIceFluxes atm_ice_fluxes(const AtmosphereAtIce &a, double ts) {
  using namespace constants;
  const double vmag = std::max(umin, std::sqrt(a.u * a.u + a.v * a.v));
  const double thvbot = a.ptem * (1.0 + zvir * a.shum);
  const double ssq = qsat(ts) / a.dens;
  const double delt = a.ptem - ts;
  const double delq = a.shum - ssq;
  const double alz = std::log(a.z / zref);
  const double cp = cpdair * (1.0 + cpvir * ssq);
  const double ltheat = latvap + latice;

  // Neutral coefficients, z/L = 0.
  const double rdn = karman / std::log(zref / zzsice);
  const double rhn = rdn;
  const double ren = rdn;
  double ustar = rdn * vmag;
  double tstar = rhn * delt;
  double qstar = ren * delq;

  double rh = rhn;
  double stable = 0.0;
  for (int iteration = 0; iteration < 2; ++iteration) {
    double hol = karman * g * a.z *
                 (tstar / thvbot + qstar / (1.0 / zvir + a.shum)) /
                 (ustar * ustar);
    hol = std::copysign(std::min(std::abs(hol), 10.0), hol);
    // Fortran's sign(0.5, hol) keeps the sign of a negative zero.
    stable = std::signbit(hol) ? 0.0 : 1.0;
    const double xsq = std::max(std::sqrt(std::abs(1.0 - 16.0 * hol)), 1.0);
    const double xqq = std::sqrt(xsq);
    const double psimh = -5.0 * hol * stable + (1.0 - stable) * psimhu(xqq);
    const double psixh = -5.0 * hol * stable + (1.0 - stable) * psixhu(xqq);

    // Shift the coefficients to the measurement height and stability.
    const double rd = rdn / (1.0 + rdn / karman * (alz - psimh));
    rh = rhn / (1.0 + rhn / karman * (alz - psixh));
    const double re = ren / (1.0 + ren / karman * (alz - psixh));
    ustar = rd * vmag;
    tstar = rh * delt;
    qstar = re * delq;
  }

  AtmIceFluxes f;
  const double tau = a.dens * ustar * ustar;
  f.taux = tau * a.u / vmag;
  f.tauy = tau * a.v / vmag;
  f.sen = cp * tau * tstar / ustar;
  f.lat = ltheat * tau * qstar / ustar;
  f.lwup = -stebol * std::pow(ts, 4);
  f.evap = f.lat / ltheat;

  // 2 m reference temperature and humidity.
  const double bn = karman / rdn;
  const double bh = karman / rh;
  const double ln0 = std::log(1.0 + (ztref / a.z) * (std::exp(bn) - 1.0));
  const double ln3 = std::log(1.0 + (ztref / a.z) * (std::exp(bn - bh) - 1.0));
  double fac = (ln0 - ztref / a.z * (bn - bh)) / bh * stable +
               (ln0 - ln3) / bh * (1.0 - stable);
  fac = std::clamp(fac, 0.0, 1.0);
  f.tref = ts + (a.tbot - ts) * fac;
  f.qref = a.shum - delq * fac;
  return f;
}

double prescribed_skin_temperature(double lat_deg, int ymd, int tod) {
  const double jday = coupling::julian_day_noleap(ymd, tod);
  const double september_first = coupling::julian_day_noleap(901, 0);
  const double c =
      std::cos(2.0 * std::numbers::pi * (jday - september_first) / 365.0);
  return lat_deg > 0.0 ? 260.0 + 10.0 * c : 260.0 - 10.0 * c;
}

double balanced_skin_temperature(const AtmosphereAtIce &atm,
                                 double sw_absorbed, double lwdn,
                                 double lat_deg, const SkinOptions &o) {
  const double thickness = lat_deg > 0.0 ? o.thickness_north : o.thickness_south;
  const double conductance =
      1.0 / (thickness / conductivity::ice + o.snow_depth / conductivity::snow);
  const auto residual = [&](double ts) {
    const auto f = atm_ice_fluxes(atm, ts);
    return sw_absorbed + lwdn + f.lwup + f.sen + f.lat +
           conductance * (constants::tkfrzsw - ts);
  };
  // The residual falls as the skin warms: every term loses heat faster.
  double hi = tmelt;
  double r_hi = residual(hi);
  if (r_hi >= 0.0) {
    return tmelt; // melting: the excess goes into melt, which the ocean has
  }
  double lo = 150.0;
  double r_lo = residual(lo);
  if (r_lo <= 0.0) {
    return lo;
  }
  double ts = std::clamp(atm.tbot, lo + 1.0, hi - 1.0);
  for (int iteration = 0; iteration < 60; ++iteration) {
    const double r = residual(ts);
    if (r > 0.0) {
      lo = ts;
    } else {
      hi = ts;
    }
    const double slope = (residual(ts + 1e-3) - r) / 1e-3;
    double next = slope < 0.0 ? ts - r / slope : 0.5 * (lo + hi);
    if (!(next > lo && next < hi)) {
      next = 0.5 * (lo + hi);
    }
    if (std::abs(next - ts) < 1e-4 || hi - lo < 1e-4) {
      return next;
    }
    ts = next;
  }
  return ts;
}

const std::vector<std::string> &sea_ice_import_names() {
  static const std::vector<std::string> names{
      "Sa_z",       "Sa_u",       "Sa_v",       "Sa_ptem",
      "Sa_shum",    "Sa_dens",    "Sa_tbot",    "Faxa_swvdr",
      "Faxa_swndr", "Faxa_swvdf", "Faxa_swndf"};
  return names;
}

const std::vector<std::string> &sea_ice_export_names() {
  static const std::vector<std::string> names{
      "Si_ifrac",   "Si_t",       "Si_tref",    "Si_qref",   "Si_snowh",
      "Si_avsdr",   "Si_anidr",   "Si_avsdf",   "Si_anidf",  "Faii_swnet",
      "Faii_sen",   "Faii_lat",   "Faii_lwup",  "Faii_evap", "Faii_taux",
      "Faii_tauy",  "Fioi_melth", "Fioi_meltw", "Fioi_salt", "Fioi_swpen",
      "Fioi_taux",  "Fioi_tauy"};
  return names;
}

SeaIceCounts compute_sea_ice_exports(coupling::ModelTime now,
                                     const SeaIceCells &cells,
                                     const fields::FieldSet &imports,
                                     fields::FieldSet &exports,
                                     const SkinOptions &skin) {
  const std::size_t n = cells.lat.size();
  if (cells.domain_mask.size() != n || cells.ice_fraction.size() != n ||
      imports.npoints() != n || exports.npoints() != n) {
    throw std::invalid_argument(
        "compute_sea_ice_exports: the cells, imports and exports disagree on "
        "the number of points.");
  }
  // Every export is written on every call, so none can be left stale.
  const auto out = [&](const char *name) {
    return exports.contains(name) ? exports.get(name) : std::span<double>{};
  };
  const auto put = [](std::span<double> f, std::size_t i, double v) {
    if (!f.empty()) {
      f[i] = v;
    }
  };
  const auto ifrac = out("Si_ifrac"), t = out("Si_t"), tref = out("Si_tref"),
             qref = out("Si_qref"), snowh = out("Si_snowh"),
             avsdr = out("Si_avsdr"), anidr = out("Si_anidr"),
             avsdf = out("Si_avsdf"), anidf = out("Si_anidf"),
             swnet = out("Faii_swnet"), sen = out("Faii_sen"),
             lat = out("Faii_lat"), lwup = out("Faii_lwup"),
             evap = out("Faii_evap"), taux = out("Faii_taux"),
             tauy = out("Faii_tauy"), melth = out("Fioi_melth"),
             meltw = out("Fioi_meltw"), salt = out("Fioi_salt"),
             swpen = out("Fioi_swpen"), otaux = out("Fioi_taux"),
             otauy = out("Fioi_tauy");
  const auto z = imports.get("Sa_z"), u = imports.get("Sa_u"),
             v = imports.get("Sa_v"), ptem = imports.get("Sa_ptem"),
             shum = imports.get("Sa_shum"), dens = imports.get("Sa_dens"),
             tbot = imports.get("Sa_tbot"), swvdr = imports.get("Faxa_swvdr"),
             swndr = imports.get("Faxa_swndr"),
             swvdf = imports.get("Faxa_swvdf"),
             swndf = imports.get("Faxa_swndf");
  const bool balance = skin.mode == SkinOptions::Mode::EnergyBalance;
  if (balance && !imports.contains("Faxa_lwdn")) {
    throw std::invalid_argument(
        "compute_sea_ice_exports: the energy-balance skin needs Faxa_lwdn.");
  }
  const auto lwdn = balance ? imports.get("Faxa_lwdn") : std::span<const double>{};

  SeaIceCounts counts;
  for (std::size_t i = 0; i < n; ++i) {
    const bool in_domain = cells.domain_mask[i] == 1.0;
    const double fraction =
        in_domain ? std::clamp(cells.ice_fraction[i], 0.0, 1.0) : 0.0;
    double ts = in_domain
                    ? prescribed_skin_temperature(cells.lat[i], now.ymd, now.tod)
                    : constants::tkfrzsw;
    AtmIceFluxes f;
    f.tref = ts;
    double sw = 0.0;
    if (in_domain) {
      ++counts.domain;
      counts.with_ice += fraction > 0.0;
      sw = (1.0 - albedo::vsdr) * swvdr[i] + (1.0 - albedo::nidr) * swndr[i] +
           (1.0 - albedo::vsdf) * swvdf[i] + (1.0 - albedo::nidf) * swndf[i];
      const AtmosphereAtIce atm{z[i],    u[i],    v[i],   ptem[i],
                                shum[i], dens[i], tbot[i]};
      if (bulk_fluxes_defined(atm)) {
        if (balance && fraction > 0.0) {
          ts = balanced_skin_temperature(atm, sw, lwdn[i], cells.lat[i], skin);
        }
        f = atm_ice_fluxes(atm, ts);
        ++counts.fluxes;
      } else {
        f.tref = ts;
        ++counts.no_atmosphere;
      }
    }
    put(ifrac, i, fraction);
    put(t, i, ts);
    put(tref, i, f.tref);
    put(qref, i, f.qref);
    put(snowh, i, nominal_snow_depth * fraction);
    put(avsdr, i, in_domain ? albedo::vsdr : 0.0);
    put(anidr, i, in_domain ? albedo::nidr : 0.0);
    put(avsdf, i, in_domain ? albedo::vsdf : 0.0);
    put(anidf, i, in_domain ? albedo::nidf : 0.0);
    put(swnet, i, sw);
    put(sen, i, f.sen);
    put(lat, i, f.lat);
    put(lwup, i, f.lwup);
    put(evap, i, f.evap);
    put(taux, i, f.taux);
    put(tauy, i, f.tauy);
    put(melth, i, 0.0);
    put(meltw, i, 0.0);
    put(salt, i, 0.0);
    put(swpen, i, 0.0);
    put(otaux, i, f.taux);
    put(otauy, i, f.tauy);
  }

  for (const auto &name : sea_ice_export_names()) {
    if (!exports.contains(name)) {
      continue;
    }
    const auto f = exports.get(name);
    const auto bad = std::count_if(f.begin(), f.end(),
                                   [](double x) { return !std::isfinite(x); });
    if (bad > 0) {
      throw std::runtime_error(
          "Sea ice export " + name + " has " + std::to_string(bad) +
          " non-finite values at " + now.to_string() +
          ". A NaN survives the coupler's ifrac weighting into every "
          "component that reads the merge.");
    }
  }
  return counts;
}

} // namespace ice
} // namespace emulator
