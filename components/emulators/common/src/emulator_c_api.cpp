#include "emulator_c_api.hpp"
#include "emulator.hpp"
#include "emulator_registry.hpp"

#include <mpi.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

/**
 * Run `body`, and turn any exception into a message and an abort.
 *
 * Every function here is called from Fortran.  An exception unwinding
 * through Fortran frames is undefined behaviour -- in practice a bare
 * `terminate` with no message, from a rank nobody is watching -- so it
 * stops at this boundary, says which call failed and why, and takes the
 * job down the way the rest of E3SM does.
 */
template <typename Body>
auto guarded(const char *where, emulator::Emulator *emu, Body &&body)
    -> decltype(body()) {
  try {
    return body();
  } catch (const std::exception &e) {
    std::cerr << "ERROR in " << where;
    if (emu) {
      std::cerr << " for emulator '" << emu->name() << "'";
    }
    std::cerr << ":\n  " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "ERROR in " << where << ": unknown exception" << std::endl;
  }
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::abort();
}

}

extern "C" {

void emulator_set_grid_data(void* handle,
                            const EmulatorGridDesc* grid) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  guarded("emulator_set_grid_data", emu, [&] { emu->set_grid_data(*grid); });
}

void emulator_setup_coupling(void* handle,
                             EmulatorCouplingDesc* cpl) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  guarded("emulator_setup_coupling", emu, [&] { emu->setup_coupling(*cpl); });
}

void emulator_init(void* handle) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  guarded("emulator_init", emu, [&] { emu->initialize(); });
}

void emulator_run(void* handle, int dt) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  guarded("emulator_run", emu, [&] { emu->run(dt); });
}

void emulator_run_at(void* handle, int dt, int ymd, int tod) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  guarded("emulator_run_at", emu,
          [&] { emu->run(dt, emulator::coupling::ModelTime{ymd, tod}); });
}

void emulator_finalize(void* handle) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  guarded("emulator_finalize", emu, [&] { emu->finalize(); });
}

void emulator_print_info(void *handle){
  auto* emu = static_cast<emulator::Emulator*>(handle);
  emu->print_info(std::cout);
}

void emulator_destroy(void* handle) {
  if (!handle) return;
  auto* emu = static_cast<emulator::Emulator*>(handle);
  // Components made by emulator_create_<kind> are not registered, so this
  // deletes them; one a caller put in the EmulatorRegistry is dropped there.
  if (!emulator::EmulatorRegistry::instance().remove_by_name(emu->name())) {
    delete emu;
  }
}

void emulator_init_coupling_indices(void* handle, const char* import_fields, const char* export_fields){
  auto* emu = static_cast<emulator::Emulator*>(handle);
  // Colon-separated coupler lists, e.g. seq_flds_x2a_fields and
  // seq_flds_a2x_fields.  Import first, as in this function's signature;
  // the list parsing (and its checks) live in fields::FieldList.
  guarded("emulator_init_coupling_indices", emu, [&] {
    emu->set_coupler_field_lists(import_fields ? import_fields : "",
                                 export_fields ? export_fields : "");
  });
}

int emulator_get_num_local_cols(void* handle) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  return emu->get_num_local_cols();
}

int emulator_get_num_global_cols(void* handle) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  return emu->get_num_global_cols();
}

int emulator_get_nx(void* handle) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  return emu->get_nx();
}

int emulator_get_ny(void* handle) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  return emu->get_ny();
}

void emulator_get_local_col_gids(void* handle, int* gids) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  emu->get_local_col_gids(gids);
}

void emulator_get_cols_latlon(void* handle, double* lat, double* lon) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  emu->get_cols_latlon(lat, lon);
}

void emulator_get_cols_area(void* handle, double* area) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  emu->get_cols_area(area);
}

void emulator_get_cols_mask_frac(void* handle, double* mask, double* frac) {
  auto* emu = static_cast<emulator::Emulator*>(handle);
  emu->get_cols_mask_frac(mask, frac);
}

} // extern "C"
