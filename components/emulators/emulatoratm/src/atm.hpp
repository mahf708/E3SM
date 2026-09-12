/**
 * @file atm.hpp
 * @brief Atmosphere emulator component declaration.
 *
 * Defines the EmulatorAtm class which implements an AI-based atmosphere
 * component for E3SM. Inherits from the Emulator base class and adds
 * atmosphere-specific coupling, field management, and inference.
 */

#ifndef EMULATORATM_HPP
#define EMULATORATM_HPP

#include "component_settings.hpp"
#include "emulator.hpp"
#include "emulator_c_api.hpp"
#include "horizontal_grid.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace emulator {

namespace atm {
class AceAtmosphere;
}

/**
 * @brief Atmosphere emulator component.
 *
 * Derived from Emulator, provides atmosphere-specific functionality:
 * - Coupling field mappings (x2a inputs, a2x outputs)
 * - AI model integration via configurable inference backends
 * - MCT interface for CIME integration
 *
 * The grid comes from the SCRIP file named by `grid_file` in atm_in, split
 * into contiguous blocks over the component's ranks.  The atmosphere covers
 * every cell, so its domain mask and frac are 1.
 *
 * ## Lifecycle
 * 1. Constructor creates EmulatorAtm with ATM_COMP type
 * 2. create_instance() sets MPI and comp_id, reads atm_in, and loads the
 *    grid from `grid_file` (or a caller provides it with set_grid_data())
 * 3. set_coupler_field_lists() and setup_coupling() (both in Emulator) bind
 *    the coupler's attribute vectors to this component's fields
 * 4. initialize() loads model and reads initial conditions
 * 5. run() executes time steps (import -> inference -> export)
 * 6. finalize() cleans up resources
 */
class EmulatorAtm : public Emulator {
public:
  EmulatorAtm();
  ~EmulatorAtm() override = default;

  // =========================================================================
  // Setup methods (called before initialize)
  // =========================================================================

  /**
   * @brief Set MPI communicator, component ID and run settings, and read
   *        atm_in.
   *
   * atm_in is `key: value` lines:
   *
   *  - `grid_file`   SCRIP file, read and split over the ranks.  `nx`/`ny`
   *                  without one are refused: dimensions with no coordinates
   *                  put every column at latitude 0.
   *  - `emulator`    ACE layout name (ACE2-EAMv3, SamudrACE-E3SMv3).  Without
   *                  it the component exchanges nothing and runs no model.
   *  - `model_path`, `ic_file`   traced checkpoint and initial condition
   *  - `device` (cuda), `dtype` (float32), `jit_optimize` (false), `seed`
   *  - `coupler_dt`  seconds; the component refuses any other dt
   *  - `surface_layer`  near_surface or lowest_level; defaults to
   *                  near_surface when the layout has the 2 m / 10 m channels
   *  - `orbit_eccen`, `orbit_obliq`, `orbit_mvelp`  orbital elements
   *                  (degrees), until the cap passes the driver's own
   */
  void create_instance(int comm, int comp_id,
                       const std::string &input_file,
                       const std::string &log_file,
                       int run_type, int start_ymd, int start_tod);


protected:
  CouplingFields coupling_fields() const override;

  // Virtual methods from Emulator base
  void init_impl() override;
  void run_impl(int dt) override;
  void final_impl() override;
  void print_extra_info(std::ostream& os) const override {};

private:
  // =========================================================================
  // Configuration
  // =========================================================================
  int m_comm = 0;              ///< MPI communicator
  std::string m_input_file;    ///< Path to atm_in config file
  std::string m_log_file;      ///< Path to log file
  int m_run_type = 0;          ///< Run type (startup/continue/branch)
  ComponentSettings m_settings; ///< atm_in, parsed
  grid::HorizontalGrid m_grid;
  grid::Decomposition m_decomp;

  /// The emulated atmosphere, once initialized; null without `emulator`.
  std::shared_ptr<atm::AceAtmosphere> m_ace;

  std::string setting(const std::string &key, const std::string &fallback) const;
};

} // namespace emulator

#endif // EMULATORATM_HPP
