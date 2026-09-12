/**
 * @file ace_atmosphere.hpp
 * @brief The ACE atmosphere, stepped by the coupler: everything EmulatorAtm
 *        does between receiving imports and handing back exports.
 */

#ifndef EMULATORATM_ACE_ATMOSPHERE_HPP
#define EMULATORATM_ACE_ATMOSPHERE_HPP

#include <mpi.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ace_surface.hpp"
#include "channel_layout.hpp"
#include "exchange.hpp"
#include "field_set.hpp"
#include "global_gather.hpp"
#include "grid_field_reader.hpp"
#include "horizontal_grid.hpp"
#include "inference_backend.hpp"
#include "insolation.hpp"
#include "interval_state.hpp"
#include "long_step_clock.hpp"
#include "network_stepper.hpp"
#include "restart_store.hpp"

namespace emulator {
namespace atm {

/// The flux channels an emulated ocean is forced with, published as
/// `atm.<name>` when Config::publish_ocean_forcing is set.
const std::vector<std::string> &ocean_forcing_channels();

/// The coupler fields the ACE atmosphere reads.
const std::vector<std::string> &ace_import_names();
/// The coupler fields it sets; every other a2x field is zeroed by the binding.
const std::vector<std::string> &ace_export_names();

/**
 * @brief One ACE atmosphere on one rank of its communicator.
 *
 * Per coupler step, run() does what EATM's run did, in the order its review
 * settled on:
 *
 *  - on the step that closes an emulator interval: surface inputs from the
 *    coupler (land deficit, fraction-weighted TS), SOLIN as the mean over
 *    the step about to be predicted, one network step with the counted step
 *    index, and the brackets advanced;
 *  - every step: snapshot channels blended, mean channels held, and the
 *    exports computed from them with the shortwave on the diurnal cycle.
 *
 * A second call at the same model time recomputes the same exports and
 * changes nothing.  initialize() takes the first step from the initial
 * condition, as EATM did, so the coupler has a state from time zero.
 */
class AceAtmosphere {
public:
  struct Config {
    fields::ChannelLayout layout;
    SurfaceOptions surface;
    Orbit orbit;
    int coupler_dt = 1800;
    /// In-process exchange with an emulated ocean; null for none.
    coupling::Exchange *exchange = nullptr;
    /**
     * Publish the ten flux channels an emulated ocean is forced with, as
     * `atm.<channel>`, every coupler step: SamudrACE's coupling, which drives
     * the ocean with the atmosphere's own fluxes rather than the coupler's
     * bulk-formula ones.  Precipitation is clipped at zero, and frozen
     * precipitation put in kg/m2/s.  Needs a layout that has all ten.
     */
    bool publish_ocean_forcing = false;
    /// At each network step, take TS and the ice/open-water split from the
    /// emulated ocean's `ocn.sst` and `ocn.sea_ice_fraction`.
    bool surface_from_ocean = false;
  };

  /**
   * @param backend the network, on the root rank of `comm`; null elsewhere
   * @param grid    the whole grid, identical on every rank
   */
  AceAtmosphere(Config config, MPI_Comm comm, const grid::HorizontalGrid &grid,
                const grid::Decomposition &decomp,
                std::shared_ptr<inference::InferenceBackend> backend);

  /**
   * @brief Start from an initial condition: every input channel, whole-grid.
   *
   * Fraction channels may hold NaN or fill values, which become 0; any other
   * unusable value is refused, naming the channel.
   */
  void initialize(coupling::ModelTime start,
                  const std::vector<grid::GridField> &initial_condition);

  /// The exports at the start time, for the coupler before the first run.
  void initial_exports(coupling::ModelTime start, fields::FieldSet &exports);

  /**
   * @param imports the ace_import_names() fields on this rank's cells
   * @param exports receives the ace_export_names() fields
   */
  void run(coupling::ModelTime now, const fields::FieldSet &imports,
           fields::FieldSet &exports);

  /// Clock, both brackets and the window insolation the held shortwave is
  /// scaled by.
  void save_to(coupling::RestartStore &store) const;
  /// Restores what save_to() wrote; boundary channels (PHIS) come from the
  /// initial condition, on every run, cold or warm.
  void restart(coupling::RestartStore &store,
               const std::vector<grid::GridField> &initial_condition);

  const coupling::LongStepClock &clock() const { return m_clock; }
  const fields::FieldSet &blended() const { return m_blended; }

private:
  void set_boundary_and_initial(const std::vector<grid::GridField> &ic,
                                bool prognostic_too);
  void compute_exports(coupling::ModelTime now, double fraction,
                       fields::FieldSet &exports);

  Config m_config;
  MPI_Comm m_comm;
  grid::Decomposition m_decomp;
  std::vector<double> m_lat, m_lon;
  grid::GlobalGather m_gather;
  coupling::NetworkStepper m_stepper;
  coupling::LongStepClock m_clock;
  coupling::BracketedState m_brackets;
  fields::FieldSet m_blended;
  Insolation m_sun;
  std::vector<double> m_solin_window;
  std::vector<double> m_solin_now;
  bool m_started = false;
};

} // namespace atm
} // namespace emulator

#endif // EMULATORATM_ACE_ATMOSPHERE_HPP
