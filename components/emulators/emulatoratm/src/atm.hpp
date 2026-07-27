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

#include "emulator.hpp"
#include "emulator_c_api.hpp"
#include "inference/create_inference_backend.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace emulator {

/**
 * @brief Atmosphere emulator component.
 *
 * Derived from Emulator, provides atmosphere-specific functionality:
 * - Coupling field mappings (x2a inputs, a2x outputs)
 * - AI model integration via configurable inference backends
 * - MCT interface for CIME integration
 *
 * Currently assumes a structured lat-lon grid. Grid dimensions (nx, ny)
 * are read from atm_in and used to compute the total global column count
 * as nx * ny. Lat/lon coordinates are stored and passed to MCT in degrees.
 *
 * ## Lifecycle
 * 1. Constructor creates EmulatorAtm with ATM_COMP type
 * 2. create_instance() sets MPI, comp_id, parses config for grid dims
 * 3. set_grid_data() sets spatial decomposition (optional override)
 * 4. init_coupling_indices() parses MCT field lists
 * 5. setup_coupling() sets buffer pointers
 * 6. initialize() loads model and reads initial conditions
 * 7. run() executes time steps (import -> inference -> export)
 * 8. finalize() cleans up resources
 */
class EmulatorAtm : public Emulator {
public:
  EmulatorAtm();
  ~EmulatorAtm() override = default;

  // =========================================================================
  // Setup methods (called before initialize)
  // =========================================================================

  /**
   * @brief Set MPI communicator, component ID, and run settings.
   */
  void create_instance(int comm, int comp_id,
                       const std::string &input_file,
                       const std::string &log_file,
                       int run_type, int start_ymd, int start_tod);

  /**
   * @brief Set grid decomposition data from driver.
   */
  void set_grid_data(const EmulatorGridDesc& grid) override;

  /**
   * @brief Initialize coupling field indices from MCT field lists.
   */
  void init_coupling_indices(const std::string &export_fields,
                             const std::string &import_fields) override;

  /**
   * @brief Set up coupling buffer pointers from MCT.
   */
  void setup_coupling(const EmulatorCouplingDesc& cpl) override;

  // =========================================================================
  // Accessors
  // =========================================================================

  int get_num_local_cols() const override { return m_num_local_cols; }
  int get_num_global_cols() const override { return m_num_global_cols; }
  int get_nx() const override { return m_nx; }
  int get_ny() const override { return m_ny; }
  void get_local_col_gids(int *gids) const override;
  void get_cols_latlon(double *lat, double *lon) const override;
  void get_cols_area(double *area) const override;

protected:
  // Virtual methods from Emulator base
  void init_impl() override;
  void run_impl(int dt) override;
  void final_impl() override;
  void print_extra_info(std::ostream& os) const override {};

private:
  // =========================================================================
  // Grid and decomposition
  // =========================================================================
  int m_nx = 0;                ///< Grid x-dimension
  int m_ny = 0;                ///< Grid y-dimension
  int m_num_local_cols = 0;    ///< Local columns on this rank
  int m_num_global_cols = 0;   ///< Total global columns
  std::vector<int> m_col_gids; ///< Global IDs for local columns
  std::vector<double> m_lat;   ///< Latitude [degrees]
  std::vector<double> m_lon;   ///< Longitude [degrees]
  std::vector<double> m_area;  ///< Cell areas

  // =========================================================================
  // Coupling
  // =========================================================================
  double *m_import_data = nullptr; ///< MCT import buffer (x2a%rAttr)
  double *m_export_data = nullptr; ///< MCT export buffer (a2x%rAttr)
  int m_num_imports = 0;           ///< Number of import fields
  int m_num_exports = 0;           ///< Number of export fields
  int m_field_size = 0;            ///< Columns per coupling field

  /// MCT field name -> its row in x2a%rAttr.
  std::map<std::string, int> m_import_idx;
  /// MCT field name -> its row in a2x%rAttr.
  std::map<std::string, int> m_export_idx;

  // =========================================================================
  // Configuration
  // =========================================================================
  int m_comm = 0;              ///< MPI communicator
  std::string m_input_file;    ///< Path to atm_in config file
  std::string m_log_file;      ///< Path to log file
  int m_run_type = 0;          ///< Run type (startup/continue/branch)

  // =========================================================================
  // Inference
  //
  // The backend is built in init_impl() from the `inference.*` settings in
  // atm_in, and handed the component communicator together with the columns
  // the coupler assigned to this rank.  With no such settings the default is
  // the stub backend, which runs no model and changes nothing.
  // =========================================================================
  std::shared_ptr<inference::InferenceBackend> m_inference;
  std::vector<std::string> m_infer_inputs;  ///< Fields the model consumes
  std::vector<std::string> m_infer_outputs; ///< Fields the model produces
  std::vector<double> m_infer_in;   ///< [n_in][ncol], one block per input
  std::vector<double> m_infer_out;  ///< [n_out][ncol], one block per output
  /// Row in x2a for each inference input, or -1 when the model names a field
  /// the coupler does not send.
  std::vector<int> m_input_src;
  /// Row in a2x for each inference output, or -1 when the model produces a
  /// field the coupler does not want.
  std::vector<int> m_output_dst;

  // =========================================================================
  // Helper methods
  // =========================================================================
  void run_inference();
  void validate_coupling(bool allow_unmatched_inputs);
  void import_coupling_fields();
  void export_coupling_fields();
  void prepare_inputs();
  void process_outputs();
};

} // namespace emulator

#endif // EMULATORATM_HPP
