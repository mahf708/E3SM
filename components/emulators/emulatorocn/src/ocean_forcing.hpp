/**
 * @file ocean_forcing.hpp
 * @brief The ocean emulator's ten forcing channels, built from what the
 *        coupler merged.
 */

#ifndef EMULATOROCN_OCEAN_FORCING_HPP
#define EMULATOROCN_OCEAN_FORCING_HPP

#include <span>
#include <string>
#include <vector>

#include "field_set.hpp"

namespace emulator {
namespace ocn {

/// The x2o fields the coupler forcing path reads.
const std::vector<std::string> &coupler_forcing_imports();

struct CouplerForcingOptions {
  /**
   * The coupler hands the ocean fluxes already weighted by the open-water
   * fraction; the emulator was trained on fluxes over the whole ocean cell.
   * Divide by (1 - Si_ifrac), floored at 0.01.  Without it FSDS fell 22% and
   * FLDS 28% by the second 5-day window in an icy winter.
   */
  bool unweight_by_ice_fraction = true;
  /// The same for the wind stress; off in EOCN by default.
  bool unweight_stress = false;
  /// The ocean albedo assumed when splitting net shortwave into down and up.
  double ocean_albedo = 0.06;
};

/**
 * @brief One coupler step's forcing, per cell, before averaging.
 *
 * Every channel is linear in the coupler's fields, so the mean of these over
 * the 5-day window is the Fortran's window-mean formula exactly; the two
 * precipitation channels are clipped at zero after averaging
 * (clip_after_mean).  Signs follow the checkpoint: fluxes positive upward
 * (FLUS, LHFLX, SHFLX), stress on the atmosphere.
 *
 * @param imports the coupler_forcing_imports() fields
 * @param forcing receives the ten samudra_forcing_names() fields
 */
void coupler_forcing_sample(const fields::FieldSet &imports,
                            const CouplerForcingOptions &options,
                            fields::FieldSet &forcing);

/// Clip the precipitation channels of a window mean at zero.
void clip_after_mean(fields::FieldSet &forcing);

/**
 * @brief Sea surface height slope on a structured, row-major global grid.
 *
 * EOCN's formula: centred differences, periodic in longitude, one-sided at
 * the first and last rows; dx uses max(cos lat, 1e-3); zero where the ocean
 * mask is 0.  `ssh` is the exported So_ssh, already zero on land, so coastal
 * slopes see those zeros exactly as the Fortran's did.
 *
 * @param lat_deg one latitude per cell (row-major, like everything else)
 */
void ssh_gradients(std::span<const double> ssh, std::span<const double> lat_deg,
                   std::span<const double> ocean_mask, int nx, int ny,
                   std::span<double> dhdx, std::span<double> dhdy);

} // namespace ocn
} // namespace emulator

#endif // EMULATOROCN_OCEAN_FORCING_HPP
