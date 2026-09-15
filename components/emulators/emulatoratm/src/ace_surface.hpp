/**
 * @file ace_surface.hpp
 * @brief The state and fluxes the ACE atmosphere hands the coupler, from its
 *        output channels.
 */

#ifndef EMULATORATM_ACE_SURFACE_HPP
#define EMULATORATM_ACE_SURFACE_HPP

#include <cstddef>
#include <span>
#include <string>

namespace emulator {
namespace atm {

/// E3SM's shr_const values, so the coupler and this agree to the bit.
namespace constants {
inline constexpr double g = 9.80616;                          // m/s2
inline constexpr double rgas = 6.02214e26 * 1.38065e-23;      // J/K/kmol
inline constexpr double rdair = rgas / 28.966;                // J/K/kg
inline constexpr double cpdair = 1.00464e3;                   // J/kg/K
inline constexpr double tkfrz = 273.15;                       // K
inline constexpr double rhofw = 1.000e3;                      // kg/m3
/// Interface between the lowest two layers of the 8-layer EAMv3 coarsening:
/// p = ak_bot + bk_bot * PS.
inline constexpr double ak_bot = 2328.474853515625;           // Pa
inline constexpr double bk_bot = 0.8722758889198303;
/// datm's fixed partition of downwelling shortwave into the coupler's bands.
inline constexpr double frac_swvdr = 0.28;
inline constexpr double frac_swndr = 0.31;
inline constexpr double frac_swvdf = 0.24;
inline constexpr double frac_swndf = 0.17;
} // namespace constants

/// Saturation vapour pressure [Pa]: datm's polynomial, over ice when
/// `tk_bot` is below freezing.  `tk` is clamped to +-50 C of freezing.
double saturation_vapor_pressure(double tk, double tk_bot);

/// Which level the exported atmospheric state describes.
enum class SurfaceLayer {
  /**
   * The emulator's 2 m / 10 m diagnostics (Tat2m, Qat2m, Uat10m, Vat10m) at
   * a 10 m reference height, with pbot = PS: datm's JRA convention.  The
   * lowest layer's midpoint is too high for the Monin-Obukhov similarity
   * the coupler's flux scheme applies.
   */
  NearSurface,
  /// The lowest layer (T_7, U_7, V_7, STW_7) at its hypsometric midpoint
  /// height above the surface.  For checkpoints without the diagnostics.
  LowestLevel
};

struct SurfaceOptions {
  SurfaceLayer layer = SurfaceLayer::NearSurface;
  double reference_height = 10.0; ///< m, NearSurface only
  /// Cap humidity at saturation.  STW_7 is total water and can be
  /// supersaturated; the flux scheme reads it as vapour.
  bool cap_humidity = true;
  /// The frozen-precipitation channel is m/s of water rather than kg/m2/s;
  /// verify against checkpoint metadata before enabling.
  bool frozen_precip_in_m_per_s = false;
  /**
   * Put the window-mean shortwave back on the diurnal cycle: scale by
   * instantaneous over window-mean insolation.  It only helps together
   * with the window-mean SOLIN input and held flux channels.
   */
  bool diurnal_shortwave = true;
};

/// Per-cell inputs for one coupler step, all the same length.  Optional
/// channels are empty spans.
struct SurfaceInputs {
  // state (snapshots, blended)
  std::span<const double> ps;         ///< Pa
  std::span<const double> phis;       ///< m2/s2
  std::span<const double> t_lowest;   ///< T_7, K
  std::span<const double> q_lowest;   ///< STW_7 / specific_total_water_7
  std::span<const double> u_lowest;   ///< U_7
  std::span<const double> v_lowest;   ///< V_7
  std::span<const double> t_2m;       ///< Tat2m (NearSurface)
  std::span<const double> q_2m;       ///< Qat2m
  std::span<const double> u_10m;      ///< Uat10m
  std::span<const double> v_10m;      ///< Vat10m
  // fluxes (interval means, held)
  std::span<const double> flds;       ///< W/m2
  std::span<const double> fsds;       ///< W/m2
  std::span<const double> fsus;       ///< W/m2, optional
  std::span<const double> precip;     ///< kg/m2/s
  std::span<const double> frozen_precip; ///< optional; see options for units
  // insolation, for diurnal_shortwave
  std::span<const double> solin_now;    ///< instantaneous, W/m2
  std::span<const double> solin_window; ///< mean over the emulator step, W/m2
};

/// The a2x fields this computes, all the same length as the inputs.
struct SurfaceExports {
  std::span<double> z, u, v, tbot, ptem, shum, pbot, pslv, dens, topo;
  std::span<double> lwdn, rainc, rainl, snowc, snowl;
  std::span<double> swndr, swvdr, swndf, swvdf, swnet;
};

/// How often each guard fired, for the component log.
struct SurfaceCounts {
  std::size_t negative_humidity = 0;
  std::size_t capped_humidity = 0;
  std::size_t negative_precip = 0;
  std::size_t negative_frozen = 0;
  std::size_t negative_fsds = 0;
  std::size_t negative_swnet = 0;
  double max_relative_humidity = 0.0; ///< before capping
};

/**
 * @brief Compute the exports.
 * @throws std::invalid_argument if a span has the wrong length, or the
 *         options need a channel the inputs do not have (NearSurface without
 *         the 2 m / 10 m channels; diurnal_shortwave without insolation)
 */
SurfaceCounts compute_surface_exports(const SurfaceInputs &in,
                                      const SurfaceOptions &options,
                                      SurfaceExports &out);

} // namespace atm
} // namespace emulator

#endif // EMULATORATM_ACE_SURFACE_HPP
