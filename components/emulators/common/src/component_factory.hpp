/**
 * @file component_factory.hpp
 * @brief How each component library creates its component for a cap.
 */

#ifndef E3SM_EMULATOR_COMPONENT_FACTORY_HPP
#define E3SM_EMULATOR_COMPONENT_FACTORY_HPP

#include "emulator.hpp"
#include "emulator_c_api.hpp"

#include <mpi.h>

#include <exception>
#include <iostream>
#include <memory>
#include <string>

namespace emulator {

/**
 * @brief Create and configure a component; the body of emulator_create_<kind>.
 *
 * Each component library defines its own emulator_create_<kind>, so the
 * atmosphere, ocean and sea ice link into one executable without two
 * definitions of one symbol, and without a cap silently binding to another
 * component's factory.  Configuring reads input and grid files and can
 * fail; the caller is Fortran, so an error is reported and the job aborted
 * rather than unwound.
 */
template <typename Component>
void *create_component(const char *kind, const EmulatorCreateConfig *cfg) {
  auto component = std::make_unique<Component>();
  try {
    component->create_instance(cfg->f_comm, cfg->comp_id,
                               cfg->input_file ? cfg->input_file : "",
                               cfg->log_file ? cfg->log_file : "",
                               cfg->run_type, cfg->start_ymd, cfg->start_tod);
  } catch (const std::exception &e) {
    std::cerr << "ERROR in emulator_create_" << kind << ":\n  " << e.what()
              << std::endl;
    MPI_Abort(MPI_Comm_f2c(cfg->f_comm), 1);
  }
  return static_cast<void *>(static_cast<Emulator *>(component.release()));
}

} // namespace emulator

#endif // E3SM_EMULATOR_COMPONENT_FACTORY_HPP
