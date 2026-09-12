/**
 * @file ice.hpp
 * @brief Sea ice component for an emulated ocean.
 */

#ifndef EMULATORICE_ICE_HPP
#define EMULATORICE_ICE_HPP

#include <string>

#include "emulator.hpp"
#include "exchange.hpp"
#include "sea_ice_surface.hpp"

namespace emulator {

/**
 * @brief The emulated ocean's sea ice, as the coupler's ice component.
 *
 * Not a sea ice model: the ocean emulator predicts the ice fraction, and
 * this reports it with a surface (compute_sea_ice_exports).  It has no grid,
 * no input file and no state of its own:
 *
 *  - the domain is the ocean's, taken from the exchange at creation, so the
 *    emulated ocean must be in this process, created first, on the same
 *    ranks.  The MCT driver creates ocn before ice.  Anything else is an
 *    error on every rank, never a guessed grid;
 *  - the fraction is the ocean's `ocn.sea_ice_fraction`, read every step.
 *    The driver runs ice before ocn, so it is the fraction the ocean
 *    exported one coupler step earlier: 30 minutes on a field whose
 *    emulator step is five days.
 *
 * Nothing to restart.
 */
class EmulatorIce : public Emulator {
public:
  explicit EmulatorIce(coupling::Exchange &exchange = coupling::Exchange::process());

  void create_instance(int comm, int comp_id, const std::string &input_file,
                       const std::string &log_file, int run_type,
                       int start_ymd, int start_tod);

  const ice::SeaIceCounts &last_counts() const { return m_counts; }

protected:
  CouplingFields coupling_fields() const override;
  void init_impl() override;
  void run_impl(int dt) override;
  void final_impl() override {}

private:
  void export_at(coupling::ModelTime now);

  coupling::Exchange &m_exchange;
  ice::SeaIceCounts m_counts;
};

} // namespace emulator

#endif // EMULATORICE_ICE_HPP
