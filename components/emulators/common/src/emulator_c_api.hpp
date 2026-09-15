/**
 * @file emulator_c_api.hpp
 * @brief Define structs and functions to hide all implementation details from Fortran API via opaque pointers
*/
#ifndef EMULATOR_C_API
#define EMULATOR_C_API
extern "C" {

/**
* @brief Configuration parameters for creating emulator instance.
*
* Fields: 
*  - f_comm: MPI communicator from Fortran
*  - comp_id: component id
*  - run_type: cold start, restart, etc...
*  - start_ymd: simulation start date
*  - start_tod: time of day in seconds
*  - input_file: config file (null terminated)
*  - log_file: emulator log file (null terminated)
*/
struct EmulatorCreateConfig {
  int  f_comm;
  int  comp_id;
  int  run_type;
  int  start_ymd;
  int  start_tod;
  const char* input_file;
  const char* log_file;
};

/**
 * @brief Description for the grid decomposition
 * 
 * Fields:
* - grid_type: structured/unstructured
* - nx: 
* - ny
* - num_local_cols
* - num_global_cols
* - col_gids
* - lat
* - lon
* - area
*/
struct EmulatorGridDesc {
  int grid_type;
  int nx;
  int ny;
  int num_local_cols;
  int num_global_cols;
  const int*    col_gids;
  const double* lat;
  const double* lon;
  const double* area;
};

/**
 * @brief Description of import and export fields to/from the coupler
 * Fields:
 *  - import_data
 *  - export_data
 *  - num_imports
 *  - num_exports
 *  - field_size
*
*/
struct EmulatorCouplingDesc {
  double* import_data;
  double* export_data;
  int     num_imports;
  int     num_exports;
  int     field_size;
};

/// Opaque handle type in C/Fortran:
/// actually points to an EmulatorComp in C++.
///
/// Each component library defines its own creator, which its cap calls, so
/// the three link into one executable.  emulator_create(kind) dispatches over
/// them; it lives in emulator_driver, for tools and tests, and returns null
/// for an unknown kind.
void* emulator_create_atm(const EmulatorCreateConfig* cfg);
void* emulator_create(const char* kind,
                      const EmulatorCreateConfig* cfg);

void  emulator_set_grid_data(void* handle,
                             const EmulatorGridDesc* grid);

void  emulator_setup_coupling(void* handle,
                              EmulatorCouplingDesc* cpl);

void emulator_init_coupling_indices(void* handle, const char* import_fields, const char* export_fields);

void  emulator_init(void* handle);
void  emulator_run(void* handle, int dt);
/// One coupler step ending at (ymd, tod): the driver's clock, not a count.
void  emulator_run_at(void* handle, int dt, int ymd, int tod);
void  emulator_finalize(void* handle);
void  emulator_print_info(void* handle);
/// Before emulator_init: restore from this restart file (null terminated).
void  emulator_set_restart_file(void* handle, const char* path);
/// After a step: write the component's restart file.  Collective.
void  emulator_write_restart(void* handle, const char* path);

/**
 * @brief Destroy an emulator instance created by emulator_create.
 *
 * Removes the instance from the EmulatorRegistry, which drops the
 * shared_ptr reference and runs the C++ destructor.  Call this after
 * emulator_finalize (or instead of it when error-aborting).
 *
 * @param handle Opaque pointer previously returned by emulator_create.
 */
void  emulator_destroy(void* handle);

} // extern "C"
#endif
