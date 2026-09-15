/**
 * @file insolation.cpp
 * @brief Ports of shr_orb_decl and shr_orb_cosz (share/util), and a
 *        window-mean SOLIN calculation.
 */

#include "physics/insolation.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace emulator {
namespace physics {

namespace {

constexpr double pi = std::numbers::pi;

} // namespace

Orbit Orbit::from_elements(double eccen, double obliq_deg, double mvelp_deg) {
  constexpr double degrad = pi / 180.0;
  Orbit o;
  o.eccen = eccen;
  o.obliqr = obliq_deg * degrad;
  // shr_orb_params: +180 degrees because the observations are earth-centred.
  o.mvelpp = (mvelp_deg + 180.0) * degrad;
  const double e2 = eccen * eccen;
  const double e3 = e2 * eccen;
  const double beta = std::sqrt(1.0 - e2);
  o.lambm0 = 2.0 * ((0.5 * eccen + 0.125 * e3) * (1.0 + beta) *
                        std::sin(o.mvelpp) -
                    0.25 * e2 * (0.5 + beta) * std::sin(2.0 * o.mvelpp) +
                    0.125 * e3 * (1.0 / 3.0 + beta) * std::sin(3.0 * o.mvelpp));
  return o;
}

void solar_declination(double calday, const Orbit &o, double &delta,
                       double &eccf) {
  constexpr double dayspy = 365.0;
  constexpr double ve = 80.5; // calday of the vernal equinox
  const double lambm = o.lambm0 + (calday - ve) * 2.0 * pi / dayspy;
  const double lmm = lambm - o.mvelpp;
  const double sinl = std::sin(lmm);
  const double lamb =
      lambm + o.eccen * (2.0 * sinl +
                         o.eccen * (1.25 * std::sin(2.0 * lmm) +
                                    o.eccen * ((13.0 / 12.0) *
                                                   std::sin(3.0 * lmm) -
                                               0.25 * sinl)));
  const double invrho = (1.0 + o.eccen * std::cos(lamb - o.mvelpp)) /
                        (1.0 - o.eccen * o.eccen);
  delta = std::asin(std::sin(o.obliqr) * std::sin(lamb));
  eccf = invrho * invrho;
}

double cos_solar_zenith(double jday, double lat, double lon, double delta) {
  return std::sin(lat) * std::sin(delta) -
         std::cos(lat) * std::cos(delta) *
             std::cos((jday - std::floor(jday)) * 2.0 * pi + lon);
}

Insolation::Insolation(const Orbit &orbit, std::span<const double> lat_deg,
                       std::span<const double> lon_deg, double s0)
    : m_orbit(orbit), m_s0(s0) {
  if (lat_deg.size() != lon_deg.size()) {
    throw std::invalid_argument("Insolation: " +
                                std::to_string(lat_deg.size()) +
                                " latitudes for " +
                                std::to_string(lon_deg.size()) +
                                " longitudes.");
  }
  m_lat_rad.reserve(lat_deg.size());
  m_lon_rad.reserve(lon_deg.size());
  for (std::size_t i = 0; i < lat_deg.size(); ++i) {
    m_lat_rad.push_back(lat_deg[i] * pi / 180.0);
    m_lon_rad.push_back(lon_deg[i] * pi / 180.0);
  }
}

void Insolation::accumulate(double jday, double weight,
                            std::span<double> out) const {
  double delta = 0.0;
  double eccf = 0.0;
  solar_declination(jday, m_orbit, delta, eccf);
  const double scale = weight * m_s0 * eccf;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] += scale * std::max(0.0, cos_solar_zenith(jday, m_lat_rad[i],
                                                     m_lon_rad[i], delta));
  }
}

void Insolation::instantaneous(int ymd, int tod, std::span<double> out) const {
  if (out.size() != m_lat_rad.size()) {
    throw std::invalid_argument("Insolation: output of the wrong length.");
  }
  std::fill(out.begin(), out.end(), 0.0);
  accumulate(julian_day_noleap(ymd, tod), 1.0, out);
}

void Insolation::window_mean(int ymd, int tod, int dt_seconds,
                             std::span<double> out, int substeps,
                             int offset_seconds) const {
  if (out.size() != m_lat_rad.size()) {
    throw std::invalid_argument("Insolation: output of the wrong length.");
  }
  if (dt_seconds <= 0 || substeps <= 0) {
    throw std::invalid_argument("Insolation: the window and its sub-steps "
                                "must be positive.");
  }
  std::fill(out.begin(), out.end(), 0.0);
  const double jday = julian_day_noleap(ymd, tod) + offset_seconds / 86400.0;
  const double dt_days = dt_seconds / 86400.0;
  for (int m = 1; m <= substeps; ++m) {
    accumulate(jday + dt_days * (m - 0.5) / substeps, 1.0 / substeps, out);
  }
}

} // namespace physics
} // namespace emulator
