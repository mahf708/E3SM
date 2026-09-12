/**
 * @file samudra_ocean.hpp
 * @brief The Samudra ocean, stepped by the coupler.
 */

#ifndef EMULATOROCN_SAMUDRA_OCEAN_HPP
#define EMULATOROCN_SAMUDRA_OCEAN_HPP

#include <mpi.h>

#include <memory>
#include <string>
#include <vector>

#include "channel_layout.hpp"
#include "exchange.hpp"
#include "field_set.hpp"
#include "global_gather.hpp"
#include "grid_field_reader.hpp"
#include "horizontal_grid.hpp"
#include "inference_backend.hpp"
#include "interval_state.hpp"
#include "long_step_clock.hpp"
#include "network_stepper.hpp"
#include "ocean_forcing.hpp"
#include "restart_store.hpp"

namespace emulator {
namespace ocn {

/// The o2x fields the ocean sets; the binding zeroes every other one.
const std::vector<std::string> &samudra_export_names();

/**
 * @brief One Samudra ocean on one rank of its communicator.
 *
 * Per coupler step run():
 *  - samples the coupler forcing (coupler_forcing_sample) into a 5-day
 *    window mean, the current step included;
 *  - on the step that closes a window: the window mean becomes the forcing
 *    channels, the network steps with the counted index, the brackets
 *    advance, and the window restarts.  The forcing is the mean over the
 *    window that just closed, as in EOCN;
 *  - every step: exports from the blended state, bounded by the ocean mask
 *    and, for the sea-ice fraction, by its own mask.
 *
 * Masked land values of the inputs need no treatment here: the traced graph
 * fills them with each channel's training mean before normalizing.
 */
class SamudraOcean {
public:
  /// Where the ten forcing channels come from.
  enum class ForcingSource {
    Coupler,   ///< the fields the MCT coupler merged (coupler_forcing_sample)
    Atmosphere ///< an emulated atmosphere's own flux channels, `atm.<name>`
  };

  struct Config {
    fields::ChannelLayout layout;
    ForcingSource forcing_source = ForcingSource::Coupler;
    /// Needed for the Atmosphere source; when set, the ocean also publishes
    /// `ocn.sst` (the exported So_t) and `ocn.sea_ice_fraction`.
    coupling::Exchange *exchange = nullptr;
    CouplerForcingOptions forcing;
    int coupler_dt = 1800;
    double freezing_sst = 271.35; ///< K: So_t floor, and its land value
    double land_salinity = 34.7;  ///< g/kg: So_s on land
  };

  SamudraOcean(Config config, MPI_Comm comm, const grid::HorizontalGrid &grid,
               const grid::Decomposition &decomp,
               std::shared_ptr<inference::InferenceBackend> backend);

  /**
   * @brief Start from the initial condition: the input channels (without the
   *        `:next` copies) plus `mask_2d` and `mask_ocean_sea_ice_fraction`,
   *        whole-grid.  Takes the first network step.
   */
  void initialize(coupling::ModelTime start,
                  const std::vector<grid::GridField> &initial_condition);

  /// Names initialize() needs from the initial condition file.
  std::vector<std::string> initial_condition_names() const;

  void run(coupling::ModelTime now, const fields::FieldSet &imports,
           fields::FieldSet &exports);
  void initial_exports(fields::FieldSet &exports);

  /// This rank's sea-ice fraction as last exported, bounded by its mask:
  /// what the sea-ice component reports.
  std::span<const double> sea_ice_fraction() const { return m_ice_fraction; }

  const coupling::BracketedState &brackets() const { return m_brackets; }
  const coupling::LongStepClock &clock() const { return m_clock; }
  std::span<const double> ocean_mask() const { return m_ocean_mask; }
  std::span<const double> ice_mask() const { return m_ice_mask; }

  void save_to(coupling::RestartStore &store) const;
  void restart(coupling::RestartStore &store,
               const std::vector<grid::GridField> &initial_condition);

private:
  void load_static(const std::vector<grid::GridField> &ic);
  void compute_exports(double fraction, fields::FieldSet &exports);

  Config m_config;
  grid::Decomposition m_decomp;
  grid::GlobalGather m_gather;
  coupling::NetworkStepper m_stepper;
  coupling::LongStepClock m_clock;
  coupling::BracketedState m_brackets;
  coupling::IntervalMean m_window;
  fields::FieldSet m_sample;
  fields::FieldSet m_blended;
  std::vector<double> m_global_lat;  ///< root only
  std::vector<double> m_global_mask; ///< root only, filled on load
  int m_nx = 0, m_ny = 0;
  std::vector<double> m_ocean_mask;
  std::vector<double> m_ice_mask;
  std::vector<double> m_ice_fraction;
  bool m_started = false;
};

} // namespace ocn
} // namespace emulator

#endif // EMULATOROCN_SAMUDRA_OCEAN_HPP
