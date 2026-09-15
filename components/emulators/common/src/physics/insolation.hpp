/**
 * @file insolation.hpp
 * @brief Top-of-atmosphere insolation, instantaneous and as a window mean.
 */

#ifndef E3SM_EMULATOR_PHYSICS_INSOLATION_HPP
#define E3SM_EMULATOR_PHYSICS_INSOLATION_HPP

#include <span>
#include <vector>

#include "calendar.hpp"

namespace emulator {
namespace physics {

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

/// Total solar irradiance, W/m2, as RRTMG uses it.
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
 * The SOLIN channel used by emulator models is a window mean (`cell_methods:
 * "time: mean"` over the preceding window), not an instantaneous snapshot;
 * the two fields share the same global mean but differ in spatial pattern,
 * so window_mean() and instantaneous() are not interchangeable.
 */
class Insolation {
public:
  /// `s0`: total solar irradiance, W/m2.
  Insolation(const Orbit &orbit, std::span<const double> lat_deg,
             std::span<const double> lon_deg, double s0 = solar_constant);

  /// S0 eccf max(0, cos z) at this moment.
  void instantaneous(int ymd, int tod, std::span<double> out) const;

  /**
   * @brief Mean over (T + offset, T + offset + dt] by the midpoint rule on
   *        `substeps`.
   *
   * The integrand is smooth except near sunrise and sunset, so few
   * sub-steps suffice. EAM's 6-hourly SOLIN averages hourly radiation
   * calls, which puts its window half an hour after the stamp: offset
   * 1800 s.
   */
  void window_mean(int ymd, int tod, int dt_seconds, std::span<double> out,
                   int substeps = 48, int offset_seconds = 0) const;

  double s0() const { return m_s0; }

private:
  void accumulate(double jday, double weight, std::span<double> out) const;

  Orbit m_orbit;
  double m_s0;
  std::vector<double> m_lat_rad;
  std::vector<double> m_lon_rad;
};

} // namespace physics
} // namespace emulator

#endif // E3SM_EMULATOR_PHYSICS_INSOLATION_HPP
