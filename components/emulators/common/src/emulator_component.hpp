/**
 * @file emulator_component.hpp
 * @brief One E3SM component class for every emulated model: its spec and
 *        its input file say what it is.
 */

#ifndef E3SM_EMULATOR_COMPONENT_HPP
#define E3SM_EMULATOR_COMPONENT_HPP

#include <memory>
#include <string>

#include "emulated_model.hpp"
#include "emulator.hpp"
#include "exchange.hpp"
#include "horizontal_grid.hpp"
#include "yaml_config.hpp"

namespace emulator {

/**
 * @brief An emulated atmosphere, ocean, sea ice or anything else.
 *
 * The input file (atm_in, ocn_in, ice_in) is YAML:
 *
 * ```yaml
 * spec: specs/samudra-e3smv3-ocean.yaml   # relative to this file, or absolute
 * coupler_dt: 1800
 * grid:
 *   file: gaussian_180x360_latlon.scrip.nc
 *   domain: ocean_mask      # full | ocean_mask | shared
 *   mask_variable: mask_2d  # ocean_mask: a binary variable of the IC file
 *   publish_as: ocn         # share this domain with other components
 *   # shared_from: ocn      # shared: take another component's published domain
 * initial_condition: samudra_ocn_ic_0_icemask.nc
 * inference:                # a network's backend; InferenceConfig keys
 *   backend: libtorch
 *   model_path: samudra_ocn_traced_masked_cuda.pt
 *   device: cuda
 *   seed: 2026
 * ```
 *
 * With no input file (an empty path) the component is unconfigured: it has
 * no domain until set_grid_data(), and exchanges and runs nothing.  A path
 * that cannot be read is an error.
 */
class EmulatorComponent : public Emulator {
public:
  EmulatorComponent(EmulatorType type, std::string name,
                    coupling::Exchange &exchange = coupling::Exchange::process());
  ~EmulatorComponent() override;

  void create_instance(int comm, int comp_id, const std::string &input_file,
                       const std::string &log_file, int run_type,
                       int start_ymd, int start_tod);

  bool configured() const { return m_spec != nullptr; }
  /// The model, once initialized; null before, or when unconfigured.
  const model::EmulatedModel *model() const { return m_model.get(); }

protected:
  CouplingFields coupling_fields() const override;
  void init_impl() override;
  void run_impl(int dt) override;
  void final_impl() override;

private:
  void setup_grid(const config::Section &grid, const std::string &base_dir);

  coupling::Exchange &m_exchange;
  int m_comm = 0;
  std::unique_ptr<config::Section> m_input;
  std::unique_ptr<model::ModelSpec> m_spec;
  std::string m_base_dir;
  int m_coupler_dt = 0;
  grid::HorizontalGrid m_grid;
  bool m_have_grid = false;
  grid::Decomposition m_decomp;
  std::unique_ptr<model::EmulatedModel> m_model;
};

} // namespace emulator

#endif // E3SM_EMULATOR_COMPONENT_HPP
