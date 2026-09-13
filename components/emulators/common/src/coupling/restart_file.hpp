/**
 * @file restart_file.hpp
 * @brief A model's restart state in one netCDF file, whatever the ranks.
 */

#ifndef E3SM_EMULATOR_COUPLING_RESTART_FILE_HPP
#define E3SM_EMULATOR_COUPLING_RESTART_FILE_HPP

#include <mpi.h>

#include <map>
#include <string>

#include "global_gather.hpp"
#include "restart_store.hpp"

namespace emulator {
namespace coupling {

/// Whether this build can read and write restart files (needs netCDF).
bool have_restart_files();

/**
 * @brief Write every rank's state, gathered, to `path`.  Collective.
 *
 * Every array must hold one value per cell of the rank; every rank must hold
 * the same names and, for integers, the same values (checked).  Arrays
 * become double variables on dimension `cells` in the grid's order, so a
 * restart does not depend on the number of ranks that wrote it.  Integers
 * become int64 scalar variables.  `attributes` are written as global text
 * attributes.
 *
 * @throws std::runtime_error on every rank if any rank fails a check or the
 *         root cannot write
 */
void write_restart_file(const std::string &path, const MemoryRestartStore &local,
                        const grid::GlobalGather &gather, MPI_Comm comm,
                        const std::map<std::string, std::string> &attributes);

/// Read `path` into this rank's share.  Collective.
MemoryRestartStore read_restart_file(const std::string &path,
                                     const grid::GlobalGather &gather,
                                     MPI_Comm comm);

} // namespace coupling
} // namespace emulator

#endif // E3SM_EMULATOR_COUPLING_RESTART_FILE_HPP
