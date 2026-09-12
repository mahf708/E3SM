/**
 * @file ace_surface.cpp
 * @brief Implementation of the ACE atmosphere's coupler exports.
 *
 * The formulas are EATM's post-review ones (emulator_comps/eatm/src/
 * ace_comp_mod.F90 on mahf708/eocn/add-samudra), each of which was measured
 * against the EAMxx control before it was kept.  The pre-review "legacy"
 * surface -- a saturated humidity at a pressure altitude above sea level --
 * is deliberately not reproduced.
 */

#include "ace_surface.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <utility>
#include <stdexcept>

namespace emulator {
namespace atm {

double saturation_vapor_pressure(double tk, double tk_bot) {
  const double t = std::min(50.0, std::max(-50.0, tk - constants::tkfrz));
  if (tk_bot < constants::tkfrz) {
    return 100.0 * (6.109177956 +
                    t * (5.034698970e-01 +
                         t * (1.886013408e-02 +
                              t * (4.176223716e-04 +
                                   t * (5.824720280e-06 +
                                        t * (4.838803174e-08 +
                                             t * 1.838826904e-10))))));
  }
  return 100.0 * (6.107799961 +
                  t * (4.436518521e-01 +
                       t * (1.428945805e-02 +
                            t * (2.650648471e-04 +
                                 t * (3.031240396e-06 +
                                      t * (2.034080948e-08 +
                                           t * 6.136820929e-11))))));
}

namespace {

void need(std::span<const double> s, std::size_t n, const char *name) {
  if (s.size() != n) {
    throw std::invalid_argument(std::string("ACE surface: ") + name + " has " +
                                std::to_string(s.size()) + " values for " +
                                std::to_string(n) + " cells.");
  }
}

void need(std::span<double> s, std::size_t n, const char *name) {
  need(std::span<const double>(s.data(), s.size()), n, name);
}

bool optional(std::span<const double> s, std::size_t n, const char *name) {
  if (s.empty()) {
    return false;
  }
  need(s, n, name);
  return true;
}

} // namespace

SurfaceCounts compute_surface_exports(const SurfaceInputs &in,
                                      const SurfaceOptions &opt,
                                      SurfaceExports &out) {
  const std::size_t n = in.ps.size();
  need(in.phis, n, "PHIS");
  need(in.flds, n, "FLDS");
  need(in.fsds, n, "FSDS");
  need(in.precip, n, "surface_precipitation_rate");

  const bool near = opt.layer == SurfaceLayer::NearSurface;
  if (near) {
    const bool all = !in.t_2m.empty() && !in.q_2m.empty() &&
                     !in.u_10m.empty() && !in.v_10m.empty();
    if (!all) {
      throw std::invalid_argument(
          "ACE surface: the near-surface layer needs Tat2m, Qat2m, Uat10m and "
          "Vat10m, and this checkpoint does not provide all four. Use the "
          "lowest-level layer for it instead; it is not chosen silently.");
    }
    need(in.t_2m, n, "Tat2m");
    need(in.q_2m, n, "Qat2m");
    need(in.u_10m, n, "Uat10m");
    need(in.v_10m, n, "Vat10m");
  } else {
    need(in.t_lowest, n, "T_7");
    need(in.q_lowest, n, "STW_7");
    need(in.u_lowest, n, "U_7");
    need(in.v_lowest, n, "V_7");
  }
  const bool have_fsus = optional(in.fsus, n, "FSUS");
  const bool have_frozen = optional(in.frozen_precip, n,
                                    "frozen_precipitation_rate");
  if (opt.diurnal_shortwave) {
    if (in.solin_now.empty() || in.solin_window.empty()) {
      throw std::invalid_argument(
          "ACE surface: diurnal shortwave needs instantaneous and window-mean "
          "insolation.");
    }
    need(in.solin_now, n, "instantaneous SOLIN");
    need(in.solin_window, n, "window-mean SOLIN");
  }
  using Named = std::pair<std::span<double>, const char *>;
  for (auto [s, name] : std::initializer_list<Named>{
           {out.z, "Sa_z"}, {out.u, "Sa_u"}, {out.v, "Sa_v"},
        {out.tbot, "Sa_tbot"}, {out.ptem, "Sa_ptem"}, {out.shum, "Sa_shum"},
        {out.pbot, "Sa_pbot"}, {out.pslv, "Sa_pslv"}, {out.dens, "Sa_dens"},
        {out.topo, "Sa_topo"}, {out.lwdn, "Faxa_lwdn"},
        {out.rainc, "Faxa_rainc"}, {out.rainl, "Faxa_rainl"},
        {out.snowc, "Faxa_snowc"}, {out.snowl, "Faxa_snowl"},
        {out.swndr, "Faxa_swndr"}, {out.swvdr, "Faxa_swvdr"},
        {out.swndf, "Faxa_swndf"}, {out.swvdf, "Faxa_swvdf"},
        {out.swnet, "Faxa_swnet"}}) {
    need(s, n, name);
  }

  using namespace constants;
  SurfaceCounts counts;

  for (std::size_t i = 0; i < n; ++i) {
    const double ps = in.ps[i];
    double pbot, tbot, shum, z, u, v;

    if (near) {
      z = opt.reference_height;
      u = in.u_10m[i];
      v = in.v_10m[i];
      tbot = in.t_2m[i];
      counts.negative_humidity += in.q_2m[i] < 0.0 ? 1 : 0;
      shum = std::max(in.q_2m[i], 0.0);
      pbot = ps;
    } else {
      const double p_int = ak_bot + bk_bot * ps;
      pbot = 0.5 * (ps + p_int);
      tbot = in.t_lowest[i];
      u = in.u_lowest[i];
      v = in.v_lowest[i];
      counts.negative_humidity += in.q_lowest[i] < 0.0 ? 1 : 0;
      shum = std::max(in.q_lowest[i], 0.0);
      const double tv = tbot * (1.0 + 0.608 * shum);
      z = (rdair * tv / g) * std::log(ps / pbot);
    }

    if (opt.cap_humidity) {
      const double e = saturation_vapor_pressure(tbot, tbot);
      const double qsat = (0.622 * e) / std::max(pbot - 0.378 * e, 1.0);
      if (shum > qsat) {
        ++counts.capped_humidity;
        if (qsat > 0.0) {
          counts.max_relative_humidity =
              std::max(counts.max_relative_humidity, shum / qsat);
        }
        shum = qsat;
      }
    }

    out.z[i] = z;
    out.u[i] = u;
    out.v[i] = v;
    out.tbot[i] = tbot;
    out.shum[i] = shum;
    out.pbot[i] = pbot;
    out.pslv[i] = ps;
    out.topo[i] = in.phis[i] / g;
    out.ptem[i] = tbot * std::pow(ps / pbot, rdair / cpdair);
    out.dens[i] = pbot / (rdair * tbot * (1.0 + 0.608 * shum));
    out.lwdn[i] = in.flds[i];

    // Precipitation: no convective/large-scale split in the emulator.
    counts.negative_precip += in.precip[i] < 0.0 ? 1 : 0;
    const double precip = std::max(in.precip[i], 0.0);
    out.rainc[i] = 0.0;
    out.snowc[i] = 0.0;
    if (have_frozen) {
      counts.negative_frozen += in.frozen_precip[i] < 0.0 ? 1 : 0;
      double snow = std::max(in.frozen_precip[i], 0.0);
      if (opt.frozen_precip_in_m_per_s) {
        snow *= rhofw;
      }
      snow = std::min(snow, precip);
      out.snowl[i] = snow;
      out.rainl[i] = precip - snow;
    } else if (tbot < tkfrz) {
      out.snowl[i] = precip;
      out.rainl[i] = 0.0;
    } else {
      out.snowl[i] = 0.0;
      out.rainl[i] = precip;
    }

    // Shortwave: the window mean, back on the diurnal cycle.
    auto diurnal = [&](double window_mean) {
      if (!opt.diurnal_shortwave) {
        return window_mean;
      }
      if (in.solin_window[i] > 1.0) {
        return window_mean * in.solin_now[i] / in.solin_window[i];
      }
      return 0.0; // polar night: no sun in the window, none now
    };
    counts.negative_fsds += in.fsds[i] < 0.0 ? 1 : 0;
    const double down = diurnal(std::max(in.fsds[i], 0.0));
    out.swvdr[i] = down * frac_swvdr;
    out.swndr[i] = down * frac_swndr;
    out.swvdf[i] = down * frac_swvdf;
    out.swndf[i] = down * frac_swndf;
    if (have_fsus) {
      const double net = down - std::max(diurnal(in.fsus[i]), 0.0);
      counts.negative_swnet += net < 0.0 ? 1 : 0;
      out.swnet[i] = std::max(net, 0.0);
    } else {
      out.swnet[i] = down;
    }
  }
  return counts;
}

} // namespace atm
} // namespace emulator
