/**
 * @file ocn.hpp
 * @brief Ocean emulator component.
 */

#ifndef EMULATOROCN_OCN_HPP
#define EMULATOROCN_OCN_HPP

#include <memory>
#include <string>

#include "component_settings.hpp"
#include "emulator.hpp"
#include "exchange.hpp"
#include "horizontal_grid.hpp"

namespace emulator {

namespace ocn {
class SamudraOcean;
}

/**
 * @brief The Samudra ocean as an E3SM component.
 *
 * ocn_in is `key: value` lines:
 *
 *  - `grid_file`  SCRIP file, split over the ranks in contiguous blocks
 *  - `ic_file`    the initial condition; its `mask_2d` is the domain mask
 *  - `emulator`   layout name (Samudra-E3SMv3).  Without it the component
 *                 still reports its domain but exchanges and runs nothing.
 *  - `model_path`, `device` (cuda), `dtype`, `jit_optimize`, `seed`
 *  - `coupler_dt` seconds; the component refuses any other dt
 *  - `forcing`    `coupler` (the merged x2o fields, default) or
 *                 `atmosphere` (an emulated atmosphere's own flux channels,
 *                 through the exchange)
 *
 * The domain is published to the exchange as soon as it is known, and the
 * sea-surface temperature and ice fraction after every export, so the sea
 * ice component can take both from here.
 */
class EmulatorOcn : public Emulator {
public:
  explicit EmulatorOcn(coupling::Exchange &exchange = coupling::Exchange::process());
  ~EmulatorOcn() override;

  void create_instance(int comm, int comp_id, const std::string &input_file,
                       const std::string &log_file, int run_type,
                       int start_ymd, int start_tod);

  const ocn::SamudraOcean *model() const { return m_ocean.get(); }

protected:
  CouplingFields coupling_fields() const override;
  void init_impl() override;
  void run_impl(int dt) override;
  void final_impl() override;

private:
  coupling::Exchange &m_exchange;
  int m_comm = 0;
  ComponentSettings m_settings;
  grid::HorizontalGrid m_grid;
  grid::Decomposition m_decomp;
  std::unique_ptr<ocn::SamudraOcean> m_ocean;
};

} // namespace emulator

#endif // EMULATOROCN_OCN_HPP
