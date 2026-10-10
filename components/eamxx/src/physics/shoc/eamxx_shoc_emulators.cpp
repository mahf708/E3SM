#include "physics/shoc/eamxx_shoc_process_interface.hpp"

#ifdef EAMXX_HAS_PROCESS_EMULATORS
#include "share/emulation/eamxx_process_emulator.hpp"
#endif

#include <ekat_assert.hpp>

#include <string>
#include <vector>

/*
 * Emulators of SHOC's eddy diffusivities.
 *
 * shoc parameters:
 *   eddy_diffusivity_emulators: [emu1, ...]   # run in this order
 *   emu1:                                     # see share/emulation/eamxx_process_emulator.hpp
 *     backend: libtorch
 *     model_path: /path/to/model.pt
 *     inputs:  [tke, shoc_mix, brunt, shoc_tabs, pblh, ...]  # SHOCEddyDiffusivityState
 *     outputs: [tk, tkh]                      # tk, tkh, tke, isotropy, or masks
 *
 * With emulators, shoc_main runs them after computing TKE and the eddy
 * diffusivities (shoc_tke), outside of any kernel, and the implicit diffusion
 * solver then uses the emulated ones. Inputs are the outputs (as shoc_tke, or
 * the emulators before, left them), and SHOC's state on midpoints, interfaces
 * (zi_grid, presi, dz_zi) and per column (pblh, obklen, ustar, dx, ...).
 * Requires SCREAM_SHOC_SMALL_KERNELS=ON.
 */

namespace scream
{

// =========================================================================================
void SHOCMacrophysics::initialize_emulators ()
{
  const auto names = m_params.isParameter("eddy_diffusivity_emulators") ?
                     m_params.get<std::vector<std::string>>("eddy_diffusivity_emulators") :
                     std::vector<std::string>();
  if (names.empty()) {
    return;
  }
#ifndef SCREAM_SHOC_SMALL_KERNELS
  EKAT_ERROR_MSG ("[SHOCMacrophysics] Error! eddy_diffusivity_emulators requires SCREAM_SHOC_SMALL_KERNELS=ON.\n"
                  "  Emulators run between the kernels of shoc_main.\n");
#elif !defined(EAMXX_HAS_PROCESS_EMULATORS)
  EKAT_ERROR_MSG ("[SHOCMacrophysics] Error! eddy_diffusivity_emulators requires EAMXX_ENABLE_PROCESS_EMULATORS=ON.\n");
#else
  for (const auto& n : names) {
    EKAT_REQUIRE_MSG (m_params.isSublist(n),
        "[SHOCMacrophysics] Error! Missing parameter sublist for emulator '" + n + "'.\n");
    auto emu = std::make_shared<ProcessEmulator>(n, m_params.sublist(n));
    m_atm_logger->info("[SHOCMacrophysics] Eddy-diffusivity emulator '" + n + "' emulates:");
    for (const auto& t : emu->target_names()) {
      EKAT_REQUIRE_MSG (t=="tk" or t=="tkh" or t=="tke" or t=="isotropy",
          "[SHOCMacrophysics] Error! Output '" + t + "' of emulator '" + n + "' is not tk, tkh, tke, "
          "isotropy, nor a mask.\n");
      m_atm_logger->info("    " + t);
    }
    m_eddy_diffusivity_emulators.push_back(emu);
  }
  m_hooks.eddy_diffusivities = [this](const SHF::SHOCEddyDiffusivityState& s) {
    run_eddy_diffusivity_emulators(s);
  };
#endif
}

#ifdef SCREAM_SHOC_SMALL_KERNELS
// =========================================================================================
void SHOCMacrophysics::run_eddy_diffusivity_emulators (const SHF::SHOCEddyDiffusivityState& s)
{
#ifdef EAMXX_HAS_PROCESS_EMULATORS
  ProcessEmulator::arrays_t inputs, targets;
  for (const auto& [name, v] : s.state) {
    inputs[name] = ProcessEmulator::array(v, s.nlev);
  }
  for (const auto& [name, v] : s.interface_state) {
    inputs[name] = ProcessEmulator::array(v, s.nlevi);
  }
  for (const auto& [name, v] : s.column_state) {
    inputs[name] = ProcessEmulator::array(v);
  }
  for (const auto& [name, v] : s.outputs) {
    inputs[name]  = ProcessEmulator::array(v, s.nlev);
    targets[name] = ProcessEmulator::array(v, s.nlev);
  }
  for (const auto& emu : m_eddy_diffusivity_emulators) {
    emu->run(inputs, targets);
  }
#endif
}
#endif

} // namespace scream
