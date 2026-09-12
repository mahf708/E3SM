/**
 * @file insolation.hpp
 * @brief Top-of-atmosphere insolation, instantaneous and as a window mean.
 */

#ifndef EMULATORATM_INSOLATION_HPP
#define EMULATORATM_INSOLATION_HPP

#include <span>
#include <vector>

#include "calendar.hpp"

namespace emulator {
namespace atm {

/// The driver's orbital parameters (seq_infodata orb_*), in its units.
struct Orbit {
  double eccen = 0.0;  ///< eccentricity
  double mvelpp = 0.0; ///< moving vernal equinox longitude of perihelion + pi, rad
  double lambm0 = 0.0; ///< mean longitude of perihelion at the vernal equinox, rad
  double obliqr = 0.0; ///< obliquity, rad

  /**
   * @brief From the orbital elements, as shr_orb_params derives them.
   * @param eccen eccentricity
   * @param obliq_deg obliquity, degrees
   * @param mvelp_deg moving vernal equinox longitude of perihelion, degrees
   */
  static Orbit from_elements(double eccen, double obliq_deg, double mvelp_deg);
};

/// Total solar irradiance, W/m2, as RRTMG and EATM use it.
inline constexpr double solar_constant = 1368.22;

using coupling::julian_day_noleap;

/// shr_orb_decl: solar declination (rad) and the earth-sun distance factor.
void solar_declination(double calday, const Orbit &orbit, double &delta,
                       double &eccf);

/// shr_orb_cosz, without its averaging options.
double cos_solar_zenith(double jday, double lat_rad, double lon_rad,
                        double delta);

/**
 * @brief Insolation on a set of cells.
 *
 * The emulator's SOLIN channel is a *window mean*: in the E3SMv3 6-hourly
 * training stream it carries `cell_methods: "time: mean"` over the six hours
 * ending at its timestamp, and fme feeds it from the step being predicted.
 * The instantaneous field has the same global mean (342 W/m2) but is a
 * different pattern -- a bullseye under the sun rather than a 90-degree
 * band -- and they differ by 330 W/m2 RMS; handing the network the
 * instantaneous field cost the Fortran emulator 14 W/m2 at the surface.
 */
class Insolation {
public:
  Insolation(const Orbit &orbit, std::span<const double> lat_deg,
             std::span<const double> lon_deg);

  /// S0 eccf max(0, cos z) at this moment.
  void instantaneous(int ymd, int tod, std::span<double> out) const;

  /**
   * @brief Mean over (T, T + dt] by the midpoint rule on `substeps`.
   *
   * 48 sub-steps for a 6 h window leave 0.03 W/m2 RMS against a 2400-point
   * reference; the integrand is smooth except at sunrise and sunset.
   */
  void window_mean(int ymd, int tod, int dt_seconds, std::span<double> out,
                   int substeps = 48) const;

private:
  void accumulate(double jday, double weight, std::span<double> out) const;

  Orbit m_orbit;
  std::vector<double> m_lat_rad;
  std::vector<double> m_lon_rad;
};

} // namespace atm
} // namespace emulator

#endif // EMULATORATM_INSOLATION_HPP
