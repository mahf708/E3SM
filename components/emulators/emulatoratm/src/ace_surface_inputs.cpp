/**
 * @file ace_surface_inputs.cpp
 * @brief Implementation of compute_surface_inputs.
 *
 * From ace_eatm_import in emulator_comps/eatm/src/ace_comp_mod.F90 on
 * mahf708/eocn/add-samudra.  Its ablation mode (the coupler's fractions and
 * a reweighted Sx_t, straight through) is not reproduced.
 */

#include "ace_surface_inputs.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace emulator {
namespace atm {

namespace {

double clip01(double x) { return std::min(std::max(x, 0.0), 1.0); }

} // namespace

SurfaceInputCounts compute_surface_inputs(const SurfaceCouplerInputs &in,
                                          SurfaceChannels &out,
                                          double tolerance) {
  const std::size_t n = in.lfrac.size();
  auto need = [n](std::size_t size, const char *name) {
    if (size != n) {
      throw std::invalid_argument(std::string("ACE surface inputs: ") + name +
                                  " has " + std::to_string(size) +
                                  " values for " + std::to_string(n) +
                                  " cells.");
    }
  };
  need(in.ofrac.size(), "Sf_ofrac");
  need(in.ifrac.size(), "Sf_ifrac");
  need(in.sx_t.size(), "Sx_t");
  need(in.ts_emulator.size(), "TS");
  need(out.landfrac.size(), "LANDFRAC");
  need(out.ocnfrac.size(), "OCNFRAC");
  need(out.icefrac.size(), "ICEFRAC");
  need(out.ts.size(), "TS out");
  const bool from_ocean = !in.ocean_ice_fraction.empty();
  if (from_ocean != !in.ocean_sst.empty()) {
    throw std::invalid_argument(
        "ACE surface inputs: an emulated ocean's ice fraction and SST come "
        "together or not at all.");
  }
  if (from_ocean) {
    need(in.ocean_ice_fraction.size(), "ocean sea-ice fraction");
    need(in.ocean_sst.size(), "ocean SST");
  }

  SurfaceInputCounts counts;
  for (std::size_t i = 0; i < n; ++i) {
    double fo = clip01(in.ofrac[i]);
    double fi = clip01(in.ifrac[i]);
    double fl = clip01(in.lfrac[i]);
    const double clip = std::max({std::abs(fo - in.ofrac[i]),
                                  std::abs(fi - in.ifrac[i]),
                                  std::abs(fl - in.lfrac[i])});
    if (clip > 0.0) {
      ++counts.clipped;
      counts.worst_clip = std::max(counts.worst_clip, clip);
    }
    double covered = fo + fi + fl;
    if (covered > 1.0) {
      ++counts.renormalized;
      counts.worst_excess = std::max(counts.worst_excess, covered - 1.0);
      fo /= covered;
      fi /= covered;
      fl /= covered;
      covered = 1.0;
    }
    const double deficit = 1.0 - covered;
    const double land = clip01(fl + deficit);

    out.landfrac[i] = fl + deficit;
    out.ocnfrac[i] = fo;
    out.icefrac[i] = fi;
    if (from_ocean) {
      const double s = clip01(in.ocean_ice_fraction[i]);
      out.icefrac[i] = s * (1.0 - land);
      out.ocnfrac[i] = std::max(1.0 - land - s * (1.0 - land), 0.0);
      const double open = std::max((1.0 - land) * (1.0 - s), 0.0);
      out.ts[i] = open * in.ocean_sst[i] + (1.0 - open) * in.ts_emulator[i];
    } else {
      out.ts[i] = in.sx_t[i] + deficit * in.ts_emulator[i];
    }
  }

  if (std::max(counts.worst_clip, counts.worst_excess) > tolerance) {
    std::ostringstream oss;
    oss << "The coupler's surface fractions are outside [0, 1] or sum above "
           "one by more than " << tolerance << " (worst clip "
        << counts.worst_clip << " in " << counts.clipped
        << " cells, worst excess " << counts.worst_excess << " in "
        << counts.renormalized
        << " cells). The merged Sx_t no longer matches the fractions the "
           "network would be given.";
    throw std::runtime_error(oss.str());
  }
  return counts;
}

} // namespace atm
} // namespace emulator
