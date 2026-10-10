#include "physics/p3/eamxx_p3_process_interface.hpp"
#include "physics/p3/p3_process_rates.hpp"

#ifdef EAMXX_HAS_PROCESS_EMULATORS
#include "share/emulation/eamxx_process_emulator.hpp"
#endif

#include <ekat_assert.hpp>

#include <algorithm>
#include <string>
#include <vector>

/*
 * Emulators of P3 process rates.
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
 * With emulators, p3_main computes the process rates, then runs the emulators
 * outside of any kernel, then applies the (partly emulated) rates. P3's
 * conservation checks, the update of the state and the diagnostics then run
 * on the emulated rates as on the stock ones.
 */

namespace scream
{

// =========================================================================================
std::vector<std::string> P3Microphysics::process_emulator_names () const
{
  return m_params.isParameter("process_emulators") ?
         m_params.get<std::vector<std::string>>("process_emulators") : std::vector<std::string>();
}

// =========================================================================================
void P3Microphysics::check_process_emulators_support () const
{
  const auto names = process_emulator_names();
  if (names.empty()) {
    return;
  }
#ifndef SCREAM_P3_SMALL_KERNELS
  EKAT_ERROR_MSG ("[P3Microphysics] Error! process_emulators requires SCREAM_P3_SMALL_KERNELS=ON.\n"
                  "  Emulators run between the kernels computing and applying the process rates.\n");
#endif
#ifndef EAMXX_HAS_PROCESS_EMULATORS
  EKAT_ERROR_MSG ("[P3Microphysics] Error! process_emulators requires EAMXX_ENABLE_PROCESS_EMULATORS=ON.\n");
#endif
  for (const auto& n : names) {
    EKAT_REQUIRE_MSG (m_params.isSublist(n),
        "[P3Microphysics] Error! Missing parameter sublist for process emulator '" + n + "'.\n");
  }
}

#if defined(SCREAM_P3_SMALL_KERNELS) && defined(EAMXX_HAS_PROCESS_EMULATORS)
// =========================================================================================
void P3Microphysics::initialize_process_emulators ()
{
  using PR = p3::P3ProcessRates;

  const auto names = process_emulator_names();
  if (names.empty()) {
    return;
  }

  bool rates_as_inputs = false;
  std::vector<int> emulated;
  for (const auto& n : names) {
    auto emu = std::make_shared<ProcessEmulator>(n, m_params.sublist(n));
    for (const auto& in : emu->input_names()) {
      rates_as_inputs |= PR::index(in)>=0;
    }
    m_atm_logger->info("[P3Microphysics] Process emulator '" + n + "' emulates:");
    for (const auto& t : emu->target_names()) {
      const int r = PR::index(t);
      EKAT_REQUIRE_MSG (r>=0, "[P3Microphysics] Error! Output '" + t + "' of process emulator '" + n +
                        "' is not a P3 process rate (see p3_process_rates.hpp), nor a mask.\n");
      emulated.push_back(r);
      m_atm_logger->info("    " + t);
    }
    m_process_emulators.push_back(emu);
  }

  const int nk_pack = ekat::npack<Pack>(m_num_levs);
  auto& hook = m_process_rates_hook;
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
#endif

} // namespace scream
