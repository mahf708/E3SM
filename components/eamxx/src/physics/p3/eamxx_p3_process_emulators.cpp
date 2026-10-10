#include "physics/p3/eamxx_p3_process_interface.hpp"
#include "physics/p3/p3_process_rates.hpp"

#ifdef EAMXX_HAS_PROCESS_EMULATORS
#include "share/emulation/eamxx_process_emulator.hpp"
#endif

#include <ekat_assert.hpp>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

/*
 * Emulators of P3 process rates, and of P3 sedimentation.
 *
 * p3 parameters:
 *   process_emulators: [emu1, emu2, ...]   # run in this order
 *   emu1:                                  # see share/emulation/eamxx_process_emulator.hpp
 *     backend: python
 *     inputs:  [qc, nc, qr, nr, rho]       # P3 state (P3ProcessState) or process rates
 *     outputs: [qc2qr_autoconv_tend, ...]  # process rates (p3_process_rates.hpp) or masks
 *     ...
 *   process_emulators_limit_self_collection: true
 *     Limit emulated nc/nr self-collection rates, so that they cannot remove more
 *     number than there is in one step (see run_process_emulators)
 *
 *   sedimentation_emulators: [sed1, ...]   # run in this order
 *   sed1:
 *     inputs:  [qr, nr, rho, dz]           # state before sedimentation, sedimentation
 *                                          # tendencies, or surface precipitation
 *     outputs: [qr_sed_tend, nr_sed_tend, precip_liq_surf]
 *     ...
 *   sedimentation_emulators_diagnose_precip: true
 *     Diagnose the surface precipitation from the column-integrated tendencies,
 *     when emulators change the qc/qr (or qi) tendencies but not precip_liq_surf
 *     (or precip_ice_surf)
 *
 * With process emulators, p3_main computes the process rates, then runs the
 * emulators outside of any kernel, then applies the (partly emulated) rates.
 * P3's conservation checks, the update of the state and the diagnostics then
 * run on the emulated rates as on the stock ones. With sedimentation
 * emulators, p3_main runs sedimentation, then the emulators, then re-applies
 * the emulated tendencies to the state before sedimentation.
 */

namespace scream
{

namespace {
const std::vector<std::string>& emulator_lists () {
  static const std::vector<std::string> lists = {"process_emulators", "sedimentation_emulators"};
  return lists;
}
} // anonymous namespace

// =========================================================================================
std::vector<std::string> P3Microphysics::emulator_names (const std::string& list) const
{
  return m_params.isParameter(list) ?
         m_params.get<std::vector<std::string>>(list) : std::vector<std::string>();
}

// =========================================================================================
void P3Microphysics::check_process_emulators_support () const
{
  for (const auto& list : emulator_lists()) {
    const auto names = emulator_names(list);
    if (names.empty()) {
      continue;
    }
#ifndef SCREAM_P3_SMALL_KERNELS
    EKAT_ERROR_MSG ("[P3Microphysics] Error! " + list + " requires SCREAM_P3_SMALL_KERNELS=ON.\n"
                    "  Emulators run between the kernels of p3_main.\n");
#endif
#ifndef EAMXX_HAS_PROCESS_EMULATORS
    EKAT_ERROR_MSG ("[P3Microphysics] Error! " + list + " requires EAMXX_ENABLE_PROCESS_EMULATORS=ON.\n");
#endif
    for (const auto& n : names) {
      EKAT_REQUIRE_MSG (m_params.isSublist(n),
          "[P3Microphysics] Error! Missing parameter sublist for emulator '" + n + "' (in " + list + ").\n");
    }
  }
}

#if defined(SCREAM_P3_SMALL_KERNELS) && defined(EAMXX_HAS_PROCESS_EMULATORS)
// =========================================================================================
void P3Microphysics::initialize_process_emulators ()
{
  using PR = p3::P3ProcessRates;
  using SR = p3::P3SedimentationRates;

  const int nk_pack = ekat::npack<Pack>(m_num_levs);

  // Create the emulators of a list. is_input(name) says whether an input is one
  // of the quantities the emulators can change; target(name) registers a target.
  // Also returns the targets of emulators that skip the physics (physics: skip),
  // which no emulator of the list may read as an input: P3 does not compute them.
  struct Created {
    std::vector<std::shared_ptr<ProcessEmulator>> emus;
    bool targets_as_inputs = false;
    std::set<std::string> skipped;
  };
  auto create = [&](const std::string& list, auto&& is_input, auto&& target) {
    Created c;
    std::set<std::string> inputs;
    for (const auto& n : emulator_names(list)) {
      auto emu = std::make_shared<ProcessEmulator>(n, m_params.sublist(n));
      for (const auto& in : emu->input_names()) {
        c.targets_as_inputs |= is_input(in);
        inputs.insert(in);
      }
      m_atm_logger->info("[P3Microphysics] Emulator '" + n + "' (" + list + ") " +
                         (emu->skips_physics() ? "replaces" : "overwrites") + ":");
      for (const auto& t : emu->target_names()) {
        target(n, t);
        if (emu->skips_physics()) c.skipped.insert(t);
        m_atm_logger->info("    " + t);
      }
      c.emus.push_back(emu);
    }
    for (const auto& t : c.skipped) {
      EKAT_REQUIRE_MSG (inputs.count(t)==0,
          "[P3Microphysics] Error! '" + t + "' is an input of an emulator in " + list + ", but an emulator\n"
          "  with physics: skip replaces it, so P3 does not compute it.\n");
    }
    return c;
  };

  // Process rates
  {
    std::vector<int> emulated;
    auto [emus, rates_as_inputs, skipped_rates] = create("process_emulators",
      [](const std::string& in) { return PR::index(in)>=0; },
      [&](const std::string& n, const std::string& t) {
        const int r = PR::index(t);
        EKAT_REQUIRE_MSG (r>=0, "[P3Microphysics] Error! Output '" + t + "' of process emulator '" + n +
                          "' is not a P3 process rate (see p3_process_rates.hpp), nor a mask.\n");
        emulated.push_back(r);
      });
    m_process_emulators = emus;
    EKAT_REQUIRE_MSG (skipped_rates.empty(),
        "[P3Microphysics] Error! physics: skip is not supported yet for process_emulators.\n");
    if (not emus.empty()) {
      auto& hook = m_hooks.process_rates;
      hook.process_rates = decltype(hook.process_rates)("p3_process_rates", m_num_cols, PR::num_rates, nk_pack);
      if (rates_as_inputs) {
        // Emulators see the rates computed by P3, whatever the emulators before them did
        m_original_process_rates = decltype(m_original_process_rates)("p3_original_process_rates",
                                                                        m_num_cols, PR::num_rates, nk_pack);
      }

      auto is_emulated = [&](const int r) {
        return std::find(emulated.begin(), emulated.end(), r)!=emulated.end();
      };
      const bool limit = m_params.get<bool>("process_emulators_limit_self_collection", true);
      m_limit_emulated_nc_selfcollect = limit and is_emulated(PR::nc_selfcollect_tend);
      m_limit_emulated_nr_selfcollect = limit and is_emulated(PR::nr_selfcollect_tend);

      hook.callback = [this](const P3F::P3ProcessState& s) {
        run_process_emulators(s);
      };
    }
  }

  // Sedimentation
  {
    int mask = 0;
    bool precip_liq = false, precip_ice = false;
    auto [emus, tends_as_inputs, skipped] = create("sedimentation_emulators",
      [](const std::string& in) { return SR::index(in)>=0; },
      [&](const std::string& n, const std::string& t) {
        const int r = SR::index(t);
        if (r>=0) {
          mask |= 1 << r;
        } else if (t=="precip_liq_surf") {
          precip_liq = true;
        } else if (t=="precip_ice_surf") {
          precip_ice = true;
        } else {
          EKAT_ERROR_MSG ("[P3Microphysics] Error! Output '" + t + "' of sedimentation emulator '" + n +
                          "' is not a sedimentation tendency (see p3_process_rates.hpp),\n"
                          "  precip_liq_surf, precip_ice_surf, nor a mask.\n");
        }
      });
    m_sedimentation_emulators = emus;
    if (not emus.empty()) {
      auto& hook = m_hooks.sedimentation;
      hook.tendencies = decltype(hook.tendencies)("p3_sed_tendencies", SR::num_rates, m_num_cols, nk_pack);
      hook.before     = decltype(hook.before)("p3_sed_before", SR::num_rates, m_num_cols, nk_pack);
      hook.apply_mask = mask;

      // Species whose sedimentation the emulators replace: all of its tendencies, or none
      for (const auto& species : std::vector<std::vector<int>>{
             {SR::qc_sed_tend, SR::nc_sed_tend}, {SR::qr_sed_tend, SR::nr_sed_tend},
             {SR::qi_sed_tend, SR::ni_sed_tend, SR::qm_sed_tend, SR::bm_sed_tend}}) {
        int n = 0;
        std::string names;
        for (int r : species) {
          n += skipped.count(SR::name(r));
          names += std::string(names.empty() ? "" : ", ") + SR::name(r);
        }
        EKAT_REQUIRE_MSG (n==0 or n==static_cast<int>(species.size()),
            "[P3Microphysics] Error! physics: skip replaces the sedimentation of a species as a whole.\n"
            "  Emulate all of " + names + " with physics: skip, or none of them.\n");
        if (n>0) {
          for (int r : species) hook.skip_mask |= 1 << r;
          m_atm_logger->info("[P3Microphysics] Sedimentation of " + names + " is replaced by emulators.");
        }
      }

      const bool diagnose = m_params.get<bool>("sedimentation_emulators_diagnose_precip", true);
      auto emulated = [&](const int r) { return (mask & (1 << r)) != 0; };
      auto replaced = [&](const int r) { return (hook.skip_mask & (1 << r)) != 0; };
      hook.diagnose_precip_liq = diagnose and not precip_liq and
                                 (emulated(SR::qc_sed_tend) or emulated(SR::qr_sed_tend));
      hook.diagnose_precip_ice = diagnose and not precip_ice and emulated(SR::qi_sed_tend);
      // Replaced species add nothing to P3's surface precipitation: it must come from somewhere
      EKAT_REQUIRE_MSG (not (replaced(SR::qc_sed_tend) or replaced(SR::qr_sed_tend)) or
                        precip_liq or hook.diagnose_precip_liq,
          "[P3Microphysics] Error! The emulators replace cloud or rain sedimentation, but nothing provides\n"
          "  precip_liq_surf: emulate it, or set sedimentation_emulators_diagnose_precip: true.\n");
      EKAT_REQUIRE_MSG (not replaced(SR::qi_sed_tend) or precip_ice or hook.diagnose_precip_ice,
          "[P3Microphysics] Error! The emulators replace ice sedimentation, but nothing provides\n"
          "  precip_ice_surf: emulate it, or set sedimentation_emulators_diagnose_precip: true.\n");
      if (tends_as_inputs) {
        m_original_sedimentation_tendencies = decltype(m_original_sedimentation_tendencies)(
            "p3_original_sed_tendencies", SR::num_rates, m_num_cols, nk_pack);
      }
      hook.callback = [this](const P3F::P3SedimentationState& s) {
        run_sedimentation_emulators(s);
      };
    }
  }
}

// =========================================================================================
void P3Microphysics::run_process_emulators (const P3F::P3ProcessState& s)
{
  using PR = p3::P3ProcessRates;
  using ExeSpace = KT::ExeSpace;

  const auto& rates = s.process_rates;
  const bool keep_original = m_original_process_rates.size()>0;
  if (keep_original) {
    Kokkos::deep_copy(m_original_process_rates, rates);
  }

  // Inputs: the state, and the rates as P3 computed them; targets: the rates
  ProcessEmulator::arrays_t inputs, targets;
  for (const auto& [name, v] : s.state) {
    inputs[name] = ProcessEmulator::array(v, s.nlev);
  }
  for (int r=0; r<PR::num_rates; ++r) {
    targets[PR::name(r)] = ProcessEmulator::array(rates, r, s.nlev);
    if (keep_original) {
      inputs[PR::name(r)] = ProcessEmulator::array(m_original_process_rates, r, s.nlev);
    }
  }
  for (const auto& emu : m_process_emulators) {
    emu->run(inputs, targets);
  }

  // Emulated self-collection rates must not remove more number than there is in one step:
  // nc_conservation uses nc + nc_selfcollect_tend*dt as the source of the other nc sinks,
  // and nr_conservation counts nc2nr_autoconv_tend (not ncautr) as the source of nr.
  const bool limit_nc = m_limit_emulated_nc_selfcollect;
  const bool limit_nr = m_limit_emulated_nr_selfcollect;
  if (limit_nc or limit_nr) {
    const auto nc = s.state.at("nc");
    const auto nr = s.state.at("nr");
    const Real inv_dt = 1 / s.dt;
    Kokkos::parallel_for("p3_emulated_rates_limiters",
      Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<2>>({0, 0}, {s.ncol, ekat::npack<Pack>(s.nlev)}),
      KOKKOS_LAMBDA (const int i, const int k) {
      if (limit_nc) {
        auto& sc = rates(i, PR::nc_selfcollect_tend, k);
        sc = max(sc, -max(nc(i,k), 0) * inv_dt);
      }
      if (limit_nr) {
        auto& sc = rates(i, PR::nr_selfcollect_tend, k);
        sc = min(sc, max(nr(i,k), 0) * inv_dt + rates(i, PR::ncautr, k));
      }
    });
    Kokkos::fence();
  }
}

// =========================================================================================
void P3Microphysics::run_sedimentation_emulators (const P3F::P3SedimentationState& s)
{
  using SR = p3::P3SedimentationRates;

  const auto& tend = s.tendencies;
  const bool keep_original = m_original_sedimentation_tendencies.size()>0;
  if (keep_original) {
    Kokkos::deep_copy(m_original_sedimentation_tendencies, tend);
  }

  // Inputs: the state before sedimentation, and the tendencies as P3 computed them;
  // targets: the tendencies and the surface precipitation
  ProcessEmulator::arrays_t inputs, targets;
  for (const auto& [name, v] : s.state) {
    inputs[name] = ProcessEmulator::array(v, s.nlev);
  }
  for (int r=0; r<SR::num_rates; ++r) {
    // (num_rates, ncol, nk_pack): slice r is a (ncol, nk_pack) view
    targets[SR::name(r)] = ProcessEmulator::array(Kokkos::subview(tend, r, Kokkos::ALL, Kokkos::ALL), s.nlev);
    if (keep_original) {
      inputs[SR::name(r)] = ProcessEmulator::array(
          Kokkos::subview(m_original_sedimentation_tendencies, r, Kokkos::ALL, Kokkos::ALL), s.nlev);
    }
  }
  for (const auto& [name, v] : {std::make_pair("precip_liq_surf", s.precip_liq_surf),
                                std::make_pair("precip_ice_surf", s.precip_ice_surf)}) {
    targets[name] = ProcessEmulator::array(v);
    inputs[name]  = ProcessEmulator::array(v); // as the emulators before left them
  }
  for (const auto& emu : m_sedimentation_emulators) {
    emu->run(inputs, targets);
  }
}
#endif

} // namespace scream
