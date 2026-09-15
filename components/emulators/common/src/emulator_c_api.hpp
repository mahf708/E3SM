/**
 * @file emulator_c_api.hpp
 * @brief Define structs and functions to hide all implementation details from Fortran API via opaque pointers
*/
#ifndef EMULATOR_C_API
#define EMULATOR_C_API
extern "C" {

/**
* @brief Configuration for creating an emulator instance.
*
* f_comm is Fortran's MPI communicator; run_type distinguishes cold start,
* restart, etc.; start_ymd/start_tod give the simulation start date and
* time of day (seconds); input_file and log_file are null-terminated paths.
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
 * @brief The grid decomposition passed from Fortran: this rank's columns,
 * their global ids and coordinates. grid_type is structured or
 * unstructured.
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
 * @brief Import and export data exchanged with the coupler: import_data and
 * export_data are flat arrays of num_imports/num_exports fields, each
 * field_size columns long.
 */
struct EmulatorCouplingDesc {
  double* import_data;
  double* export_data;
  int     num_imports;
  int     num_exports;
  int     field_size;
};

/// Opaque handle: points to an EmulatorComp in C++.
///
/// Each component library defines its own creator; emulator_create(kind)
/// dispatches over them (in emulator_driver, for tools and tests) and
/// returns null for an unknown kind.
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
