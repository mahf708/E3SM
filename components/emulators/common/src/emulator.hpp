/**
 * @file emulator.hpp
 * @brief Abstract base class for all E3SM emulators.
 */

#ifndef E3SM_EMULATOR_HPP
#define E3SM_EMULATOR_HPP

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "coupler_binding.hpp"
#include "emulator_c_api.hpp"
#include "field_list.hpp"
#include "field_set.hpp"
#include "horizontal_grid.hpp"
#include "long_step_clock.hpp"

namespace emulator {

/**
 * @brief Enumeration of emulator types in E3SM.
 */
enum class EmulatorType {
  ATM_COMP = 0, ///< Atmosphere component emulator
  OCN_COMP = 1, ///< Ocean component emulator
  ICE_COMP = 2, ///< Sea ice component emulator
  LND_COMP = 3  ///< Land component emulator
};

/**
 * @brief Abstract base class for all E3SM emulators.
 *
 * Provides the common infrastructure for emulators.
 * Derived classes implement the pure virtual methods for
 * emulator-specific behavior.
 */
class Emulator {
public:
  /**
   * @brief Construct a new Emulator.
   *
   * @param type Emulator type
   * @param id Emulator ID (-1 if unassigned)
   * @param name Emulator name (empty if unassigned)
   */
  explicit Emulator(EmulatorType type, int id = -1,
                    const std::string &name = "");
  virtual ~Emulator() = default;

  // Lifecycle methods
  void initialize();
  void run(int dt);
  /**
   * @brief Run one coupler step ending at model time `now`.
   *
   * The time is the driver's, passed through rather than counted here: the
   * driver can call run twice at one time, and a component that counted its
   * own time would then be a step ahead for the rest of the run.
   */
  void run(int dt, coupling::ModelTime now);
  /// The model time of the current run() call; unset (-1) outside one, or
  /// if the caller did not give a time.
  coupling::ModelTime current_time() const { return m_current_time; }
  /// The run's start, as the driver gave it at creation; unset if not.
  coupling::ModelTime start_time() const { return m_start_time; }
  void finalize();

  // Accessors
  EmulatorType type() const { return m_type; }
  int id() const { return m_id; }
  const std::string &name() const { return m_name; }
  bool is_initialized() const { return m_initialized; }
  int step_count() const { return m_step_count; }
  void print_info(std::ostream& os) const;

  // ---------------------------------------------------------------------------
  // Grid
  //
  // A component's cells, as the coupler sees them: set once, before coupling,
  // either by the component from its own grid file (set_domain) or by a
  // caller that already has the decomposition (set_grid_data).  The getters
  // below read it back for the cap, and a component that never set a grid
  // reports zero cells rather than inventing coordinates.
  // ---------------------------------------------------------------------------

  /// Take the decomposition from a caller; mask and frac are 1 everywhere.
  virtual void set_grid_data(const EmulatorGridDesc& grid);

  bool has_domain() const { return m_has_domain; }
  const grid::Domain &domain() const { return m_domain; }

  // ---------------------------------------------------------------------------
  // Coupling
  //
  // The base class owns the exchange with the coupler; a component only says
  // which fields it reads and writes (coupling_fields()) and then reads
  // imports() and writes mutable_exports().  The cap calls, in order:
  //
  //   set_coupler_field_lists(x2c, c2x)  the coupler's colon-separated lists
  //   setup_coupling(cpl)                the attribute vectors' buffers
  //   initialize()                       init_impl(), then exports pushed
  //   run(dt)                            imports pulled, run_impl(), exports
  //                                      pushed
  //
  // so a component never touches a coupler buffer, and every name is checked
  // once, in setup_coupling(), rather than failing (or not) on first use.
  // ---------------------------------------------------------------------------

  /// What a component reads from and writes to the coupler.
  struct CouplingFields {
    std::vector<fields::FieldSpec> imports;
    std::vector<fields::FieldSpec> exports;
  };

  /// The coupler's field lists, import (x2c) first.
  void set_coupler_field_lists(std::string_view import_fields,
                               std::string_view export_fields);

  /**
   * @brief Bind the coupler's attribute vectors to this component's fields.
   *
   * Needs set_coupler_field_lists() and set_grid_data() first.  Throws if a
   * list disagrees with its vector's field count, if the vectors' length is
   * not this rank's column count, or if coupling_fields() asks for a field
   * the coupler does not carry.
   */
  void setup_coupling(const EmulatorCouplingDesc& cpl);

  bool is_coupled() const { return m_import_binding.has_value(); }
  /// Import fields, current as of the last pull (the start of run()).
  const fields::FieldSet &imports() const { return m_imports; }
  const fields::FieldSet &exports() const { return m_exports; }
  const fields::CouplerBinding *import_binding() const {
    return m_import_binding ? &*m_import_binding : nullptr;
  }
  const fields::CouplerBinding *export_binding() const {
    return m_export_binding ? &*m_export_binding : nullptr;
  }


  // What the cap reads back to build the gsMap and the MCT domain.
  virtual int get_num_local_cols() const;
  virtual int get_num_global_cols() const;
  virtual int get_nx() const;
  virtual int get_ny() const;
  virtual void get_local_col_gids(int* gids) const;
  virtual void get_cols_latlon(double* lat, double* lon) const;
  virtual void get_cols_area(double* area) const;
  /// Mask and frac for the MCT domain; see grid::Domain for the rule.
  virtual void get_cols_mask_frac(double* mask, double* frac) const;

protected:
  void set_start_time(coupling::ModelTime start) { m_start_time = start; }

  /**
   * @brief Set this rank's cells.
   * @param nx, ny     the global logical shape (ny is 1 if unstructured)
   * @param num_global the global cell count
   * @throws std::logic_error once coupling is set up: the attribute vectors
   *         were sized from the old domain
   */
  void set_domain(grid::Domain domain, int nx, int ny, std::size_t num_global);

  /// The fields this component exchanges; none by default.
  virtual CouplingFields coupling_fields() const { return {}; }

  /// Export fields, writable: pushed to the coupler after init_impl() and
  /// after every run_impl().  (Imports are read through imports().)
  fields::FieldSet &mutable_exports() { return m_exports; }

  /// Per-channel masks named by export specs.  Set them before coupling is
  /// set up; sized to the domain when set_domain() is called.
  fields::MaskSet &masks() { return m_masks; }

  // Virtual methods for derived classes
  virtual void init_impl() = 0;
  virtual void run_impl(int dt) = 0;
  virtual void final_impl() = 0;
  virtual void print_extra_info(std::ostream& os) const {}

  EmulatorType m_type;
  int m_id;
  std::string m_name;

  bool m_initialized = false;
  int m_step_count = 0;
  coupling::ModelTime m_current_time;
  coupling::ModelTime m_start_time;

private:
  void pull_imports();
  void push_exports();

  grid::Domain m_domain;
  int m_nx = 0;
  int m_ny = 0;
  std::size_t m_num_global = 0;
  bool m_has_domain = false;

  fields::FieldList m_import_list;
  fields::FieldList m_export_list;
  bool m_have_field_lists = false;
  EmulatorCouplingDesc m_cpl{};
  fields::FieldSet m_imports;
  fields::FieldSet m_exports;
  fields::MaskSet m_masks;
  std::optional<fields::CouplerBinding> m_import_binding;
  std::optional<fields::CouplerBinding> m_export_binding;
};

} // namespace emulator

#endif // E3SM_EMULATOR_HPP
