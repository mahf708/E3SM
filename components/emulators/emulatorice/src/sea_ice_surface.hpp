/**
 * @file sea_ice_surface.hpp
 * @brief What the coupler needs from sea ice, given only its fraction.
 */

#ifndef EMULATORICE_SEA_ICE_SURFACE_HPP
#define EMULATORICE_SEA_ICE_SURFACE_HPP

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "field_set.hpp"
#include "long_step_clock.hpp"

namespace emulator {
namespace ice {

/// E3SM's shr_const values, so the coupler and this agree to the bit.
namespace constants {
inline constexpr double g = 9.80616;                     // m/s2
inline constexpr double stebol = 5.670374419e-8;         // W/m2/K4
inline constexpr double rgas = 6.02214e26 * 1.38065e-23; // J/K/kmol
inline constexpr double rdair = rgas / 28.966;           // J/K/kg
inline constexpr double rwv = rgas / 18.016;             // J/K/kg
inline constexpr double zvir = rwv / rdair - 1.0;
inline constexpr double karman = 0.4;
inline constexpr double cpdair = 1.00464e3;              // J/kg/K
inline constexpr double cpvir = 1.810e3 / cpdair - 1.0;
inline constexpr double latvap = 2.501e6;                // J/kg
inline constexpr double latice = 3.337e5;                // J/kg
inline constexpr double tkfrzsw = 273.15 - 1.8;          // K
} // namespace constants

/**
 * @brief dice's surface albedos (dice_comp_mod.F90): a snow-covered pack,
 *        climatologically averaged, 28.6% snow.  What an AMIP run's ice uses.
 */
namespace albedo {
inline constexpr double snow_fraction = 0.286;
inline constexpr double vsdr = 0.500 * (1.0 - snow_fraction) + 0.800 * snow_fraction;
inline constexpr double nidr = 0.700 * (1.0 - snow_fraction) + 0.960 * snow_fraction;
inline constexpr double vsdf = 0.500 * (1.0 - snow_fraction) + 0.700 * snow_fraction;
inline constexpr double nidf = 0.700 * (1.0 - snow_fraction) + 0.950 * snow_fraction;
} // namespace albedo

/// Nominal snow depth on the ice, m; Si_snowh is this times the fraction.
inline constexpr double nominal_snow_depth = 0.20;

/// The atmosphere's lowest level over one cell, from x2i.
struct AtmosphereAtIce {
  double z = 0;    ///< Sa_z, m
  double u = 0;    ///< Sa_u, m/s
  double v = 0;    ///< Sa_v, m/s
  double ptem = 0; ///< Sa_ptem, K
  double shum = 0; ///< Sa_shum, kg/kg
  double dens = 0; ///< Sa_dens, kg/m3
  double tbot = 0; ///< Sa_tbot, K
};

/// Atmosphere-ice fluxes, positive downward, as the coupler's Faii_* are.
struct AtmIceFluxes {
  double sen = 0;  ///< W/m2
  double lat = 0;  ///< W/m2
  double lwup = 0; ///< W/m2, negative
  double evap = 0; ///< kg/m2/s
  double taux = 0; ///< N/m2
  double tauy = 0; ///< N/m2
  double tref = 0; ///< K, 2 m
  double qref = 0; ///< kg/kg, 2 m
};

/**
 * @brief Whether the bulk scheme can be evaluated on this state.
 *
 * At initialization the coupler has not filled x2i, so the state is all
 * zeros, and the scheme divides by density and takes log(z).  A NaN from
 * that survives the merge's ifrac weighting and, in EICE, took down EAM's
 * first physics step rather than showing up as an obviously wrong flux.
 */
bool bulk_fluxes_defined(const AtmosphereAtIce &atm);

/**
 * @brief dice's atmosphere-ice bulk fluxes over a surface at `ts`: the
 *        formulae of EICE's eice_flux_atmice, one cell.
 *
 * Two fixed iterations on the stability, as dice and EICE do.  Needs
 * bulk_fluxes_defined(atm).
 */
AtmIceFluxes atm_ice_fluxes(const AtmosphereAtIce &atm, double ts);

/**
 * @brief dice's prescribed ice skin temperature: 260 K plus or minus 10 K on
 *        a seasonal cosine peaking on 1 September in the north.
 *
 * The emulated ocean carries a fraction, not an energy balance, so there is
 * nothing better to report.  It is the skin over the ice-covered part of the
 * cell and is *not* blended towards freezing as the fraction vanishes: the
 * coupler weights Si_t by ifrac on its way to the atmosphere, so blending
 * here too would scale the thermal anomaly by ifrac squared.
 */
double prescribed_skin_temperature(double lat_deg, int ymd, int tod);

/**
 * @brief How the ice reports its surface temperature.
 *
 * Prescribed is dice's seasonal skin (prescribed_skin_temperature): enough
 * for an emulated atmosphere, which never reads it.  A model atmosphere
 * does, and a skin held at 254 K under 262 K air pulls 150-200 W/m2 of
 * sensible heat out of its lowest level.  EnergyBalance solves for the skin
 * at which the surface balances: absorbed shortwave, downwelling longwave,
 * the bulk fluxes at that skin, and conduction from the ocean at freezing
 * through `snow_depth` of snow on ice `thickness_north`/`_south` thick
 * (2 m and 1 m, as CAM's prescribed ice), capped at melting.  No state: the
 * pack has no heat capacity, so the skin follows the atmosphere each step.
 */
struct SkinOptions {
  enum class Mode { Prescribed, EnergyBalance };
  Mode mode = Mode::Prescribed;
  double thickness_north = 2.0;           ///< m
  double thickness_south = 1.0;           ///< m
  double snow_depth = nominal_snow_depth; ///< m
};

/// Thermal conductivities, W/m/K (CICE's defaults).
namespace conductivity {
inline constexpr double ice = 2.03;
inline constexpr double snow = 0.31;
} // namespace conductivity

/// The melting point of the ice surface, K.
inline constexpr double tmelt = 273.15;

/**
 * @brief The skin temperature at which the surface balances:
 *
 *   sw_absorbed + lwdn + lwup(T) + sen(T) + lat(T) + C (T_freeze - T) = 0,
 *
 * with the fluxes of atm_ice_fluxes (positive downward) and C the snow and
 * ice conductance, or tmelt if the surface would be warmer.  Newton's method
 * safeguarded by bisection on [150 K, tmelt], to 1e-4 K.  Needs
 * bulk_fluxes_defined(atm).
 */
double balanced_skin_temperature(const AtmosphereAtIce &atm,
                                 double sw_absorbed, double lwdn,
                                 double lat_deg, const SkinOptions &options);

/// The x2i fields the sea ice reads.
const std::vector<std::string> &sea_ice_import_names();
/// The i2x fields the sea ice sets.  Si_snowh is not always in the coupler's
/// list; every other one is.
const std::vector<std::string> &sea_ice_export_names();

/// One rank's cells.
struct SeaIceCells {
  std::span<const double> lat;          ///< degrees
  std::span<const double> domain_mask;  ///< 0 or 1: the ocean's domain
  std::span<const double> ice_fraction; ///< the emulated ocean's, [0, 1]
};

struct SeaIceCounts {
  std::size_t domain = 0;       ///< cells in the domain
  std::size_t with_ice = 0;     ///< of those, with a fraction above zero
  std::size_t fluxes = 0;       ///< where the bulk scheme was evaluated
  std::size_t no_atmosphere = 0; ///< domain cells skipped: state not physical
};

/**
 * @brief Fill every sea_ice_export_names() field present in `exports`.
 *
 *  - Si_ifrac: the ocean's fraction, clamped to [0, 1].  Si_ifrac is a share
 *    of the ice domain's frac, and the ocean's fraction a share of the
 *    non-land area: the same quantity, so a copy, not a conversion.
 *  - Si_t: inside the domain, the skin `skin` says (prescribed, or the
 *    balanced skin where there is ice and an atmosphere); the freezing point
 *    of sea water outside it.  EnergyBalance also reads Faxa_lwdn.
 *  - albedos, and Faii_swnet from them and x2i's four shortwave bands.
 *  - Faii_* and Si_tref/Si_qref: the bulk fluxes where they are defined;
 *    elsewhere zero, with Si_tref = Si_t (dice writes spval, and 1e30 times a
 *    small ifrac is still enormous).
 *  - Fioi_taux/tauy: the atmosphere-ice stress passed through, so that after
 *    the coupler's ifrac/afrac weighting the ocean feels what the atmosphere
 *    applied; the emulated ocean was trained on whole-cell stress.
 *  - Fioi_melth, meltw, salt, swpen: zero.  The emulated ocean advances its
 *    own ice, so melt heat and water are already inside its step; handing
 *    them over again would count them twice.
 *
 * @throws std::runtime_error naming the field if any export is not finite
 */
SeaIceCounts compute_sea_ice_exports(coupling::ModelTime now,
                                     const SeaIceCells &cells,
                                     const fields::FieldSet &imports,
                                     fields::FieldSet &exports,
                                     const SkinOptions &skin = {});

} // namespace ice
} // namespace emulator

#endif // EMULATORICE_SEA_ICE_SURFACE_HPP
