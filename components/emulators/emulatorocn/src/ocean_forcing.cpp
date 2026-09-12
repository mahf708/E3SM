/**
 * @file ocean_forcing.cpp
 * @brief The coupler forcing path of emulator_comps/eocn
 *        (ocn_import_mct and samudra_comp_mod on mahf708/eocn/add-samudra).
 */

#include "ocean_forcing.hpp"

#include <algorithm>
#include <cmath>

namespace emulator {
namespace ocn {

const std::vector<std::string> &coupler_forcing_imports() {
  static const std::vector<std::string> names{
      "Foxx_taux", "Foxx_tauy", "Faxa_rain", "Faxa_snow", "Foxx_lwup",
      "Faxa_lwdn", "Foxx_swnet", "Foxx_lat", "Foxx_sen", "Si_ifrac"};
  return names;
}

void coupler_forcing_sample(const fields::FieldSet &in,
                            const CouplerForcingOptions &opt,
                            fields::FieldSet &out) {
  const auto ifrac = in.get("Si_ifrac");
  const auto taux = in.get("Foxx_taux"), tauy = in.get("Foxx_tauy");
  const auto rain = in.get("Faxa_rain"), snow = in.get("Faxa_snow");
  const auto lwup = in.get("Foxx_lwup"), lwdn = in.get("Faxa_lwdn");
  const auto swnet = in.get("Foxx_swnet");
  const auto lat = in.get("Foxx_lat"), sen = in.get("Foxx_sen");

  auto o_taux = out.get("TAUX"), o_tauy = out.get("TAUY");
  auto o_prec = out.get("surface_precipitation_rate");
  auto o_snow = out.get("frozen_precipitation_rate");
  auto o_flus = out.get("FLUS"), o_fsus = out.get("FSUS");
  auto o_flds = out.get("FLDS"), o_fsds = out.get("FSDS");
  auto o_lh = out.get("LHFLX"), o_sh = out.get("SHFLX");

  for (std::size_t i = 0; i < ifrac.size(); ++i) {
    const double w = opt.unweight_by_ice_fraction
                         ? 1.0 / std::max(1.0 - ifrac[i], 0.01)
                         : 1.0;
    const double ws = opt.unweight_stress ? w : 1.0;
    o_taux[i] = -ws * taux[i];
    o_tauy[i] = -ws * tauy[i];
    o_prec[i] = w * (rain[i] + snow[i]);
    o_snow[i] = w * snow[i];
    o_flus[i] = -w * lwup[i];
    o_flds[i] = w * lwdn[i];
    const double down = w * swnet[i] / (1.0 - opt.ocean_albedo);
    o_fsds[i] = down;
    o_fsus[i] = down - w * swnet[i];
    o_lh[i] = -w * lat[i];
    o_sh[i] = -w * sen[i];
  }
}

void ssh_gradients(std::span<const double> ssh, std::span<const double> lat,
                   std::span<const double> mask, int nx, int ny,
                   std::span<double> dhdx, std::span<double> dhdy) {
  constexpr double rearth = 6.37122e6; // SHR_CONST_REARTH, m
  constexpr double deg2rad = 3.14159265358979323846 / 180.0;
  auto at = [nx](int j, int i) { return static_cast<std::size_t>(j * nx + i); };
  for (int j = 0; j < ny; ++j) {
    const int jp = std::min(j + 1, ny - 1);
    const int jm = std::max(j - 1, 0);
    const double coslat = std::max(std::cos(lat[at(j, 0)] * deg2rad), 1.0e-3);
    const double dy = (lat[at(jp, 0)] - lat[at(jm, 0)]) * deg2rad * rearth;
    const double dx = (360.0 / nx) * 2.0 * deg2rad * rearth * coslat;
    for (int i = 0; i < nx; ++i) {
      const int ip = i + 1 < nx ? i + 1 : 0;
      const int im = i > 0 ? i - 1 : nx - 1;
      const auto k = at(j, i);
      if (mask[k] < 0.5) {
        dhdx[k] = 0.0;
        dhdy[k] = 0.0;
        continue;
      }
      dhdx[k] = (ssh[at(j, ip)] - ssh[at(j, im)]) / dx;
      dhdy[k] = std::abs(dy) > 0.0 ? (ssh[at(jp, i)] - ssh[at(jm, i)]) / dy : 0.0;
    }
  }
}

void clip_after_mean(fields::FieldSet &forcing) {
  for (const char *name :
       {"surface_precipitation_rate", "frozen_precipitation_rate"}) {
    for (auto &v : forcing.get(name)) {
      v = std::max(v, 0.0);
    }
  }
}

} // namespace ocn
} // namespace emulator
