/**
 * @file restart_file.cpp
 * @brief Restart files over serial netCDF-C, written and read on the root.
 */

#include "restart_file.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#ifdef EMULATOR_HAVE_NETCDF
#include <netcdf.h>
#endif

namespace emulator {
namespace coupling {

namespace {

/// Every rank throws `message` if any rank's flag is 0.
void agree(int ok, MPI_Comm comm, const std::string &message) {
  int all = 0;
  MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, comm);
  if (all == 0) {
    throw std::runtime_error(message);
  }
}

/// The root's error, if any, on every rank.
void share_error(std::string &error, MPI_Comm comm, int root) {
  int length = static_cast<int>(error.size());
  MPI_Bcast(&length, 1, MPI_INT, root, comm);
  error.resize(static_cast<std::size_t>(length));
  if (length > 0) {
    MPI_Bcast(error.data(), length, MPI_CHAR, root, comm);
    throw std::runtime_error(error);
  }
}

} // namespace

#ifdef EMULATOR_HAVE_NETCDF

bool have_restart_files() { return true; }

void write_restart_file(const std::string &path,
                        const MemoryRestartStore &local,
                        const grid::GlobalGather &gather, MPI_Comm comm,
                        const std::map<std::string, std::string> &attributes) {
  const bool root = gather.is_root();
  const auto &arrays = local.arrays();
  const auto &ints = local.ints();

  // The same names and integers everywhere, and a value per cell in each
  // array: otherwise the file would describe no single run.
  int ok = 1;
  for (const auto &[name, values] : arrays) {
    ok = ok && values.size() == gather.num_local();
  }
  agree(ok, comm, "Restart '" + path + "': an array does not have one value "
                  "per cell on every rank.");
  long counts[2] = {static_cast<long>(arrays.size()),
                    static_cast<long>(ints.size())};
  long lo[2], hi[2];
  MPI_Allreduce(counts, lo, 2, MPI_LONG, MPI_MIN, comm);
  MPI_Allreduce(counts, hi, 2, MPI_LONG, MPI_MAX, comm);
  agree(lo[0] == hi[0] && lo[1] == hi[1], comm,
        "Restart '" + path + "': ranks hold different restart entries.");
  std::vector<long long> values;
  for (const auto &[_, v] : ints) {
    values.push_back(v);
  }
  std::vector<long long> vmin(values.size()), vmax(values.size());
  MPI_Allreduce(values.data(), vmin.data(), static_cast<int>(values.size()),
                MPI_LONG_LONG, MPI_MIN, comm);
  MPI_Allreduce(values.data(), vmax.data(), static_cast<int>(values.size()),
                MPI_LONG_LONG, MPI_MAX, comm);
  agree(vmin == vmax, comm,
        "Restart '" + path + "': ranks disagree on an integer (a clock?).");

  // The root defines the file; every rank then gathers each array.
  std::string error;
  int ncid = -1;
  std::vector<int> array_ids;
  auto check = [&](int status, const std::string &what) {
    if (status != NC_NOERR && error.empty()) {
      error = "Restart '" + path + "': cannot " + what + ": " +
              nc_strerror(status);
    }
    return status == NC_NOERR;
  };
  if (root) {
    if (check(nc_create(path.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid),
              "create")) {
      int cells = 0;
      check(nc_def_dim(ncid, "cells", gather.num_global(), &cells),
            "define cells");
      for (const auto &[key, text] : attributes) {
        check(nc_put_att_text(ncid, NC_GLOBAL, key.c_str(), text.size(),
                              text.c_str()),
              "write attribute " + key);
      }
      for (const auto &[name, _] : arrays) {
        int id = -1;
        check(nc_def_var(ncid, name.c_str(), NC_DOUBLE, 1, &cells, &id),
              "define " + name);
        array_ids.push_back(id);
      }
      std::vector<int> int_ids;
      for (const auto &[name, _] : ints) {
        int id = -1;
        check(nc_def_var(ncid, name.c_str(), NC_INT64, 0, nullptr, &id),
              "define " + name);
        int_ids.push_back(id);
      }
      check(nc_enddef(ncid), "leave define mode");
      for (std::size_t k = 0; k < int_ids.size() && error.empty(); ++k) {
        check(nc_put_var_longlong(ncid, int_ids[k], &values[k]),
              "write an integer");
      }
    }
  }
  std::vector<double> global(root ? gather.num_global() : 0);
  std::size_t k = 0;
  for (const auto &[name, v] : arrays) {
    gather.gather(v, global);
    if (root && error.empty()) {
      check(nc_put_var_double(ncid, array_ids[k], global.data()),
            "write " + name);
    }
    ++k;
  }
  if (root && ncid >= 0) {
    check(nc_close(ncid), "close");
  }
  share_error(error, comm, 0);
}

MemoryRestartStore read_restart_file(const std::string &path,
                                     const grid::GlobalGather &gather,
                                     MPI_Comm comm) {
  const bool root = gather.is_root();
  std::string error;
  int ncid = -1;
  auto check = [&](int status, const std::string &what) {
    if (status != NC_NOERR && error.empty()) {
      error = "Restart '" + path + "': cannot " + what + ": " +
              nc_strerror(status);
    }
    return status == NC_NOERR;
  };
  // The root lists the file: array names (on `cells`) and integers.
  std::vector<std::string> array_names, int_names;
  std::vector<long long> int_values;
  if (root && check(nc_open(path.c_str(), NC_NOWRITE, &ncid), "open")) {
    int nvars = 0;
    check(nc_inq_nvars(ncid, &nvars), "count variables");
    for (int id = 0; id < nvars && error.empty(); ++id) {
      char name[NC_MAX_NAME + 1] = {0};
      int ndims = 0;
      nc_type type = NC_NAT;
      check(nc_inq_varname(ncid, id, name), "read a variable name");
      check(nc_inq_varndims(ncid, id, &ndims), "read the rank of " +
                                                   std::string(name));
      check(nc_inq_vartype(ncid, id, &type), "read the type of " +
                                                 std::string(name));
      if (ndims == 1) {
        int dim = 0;
        std::size_t len = 0;
        check(nc_inq_vardimid(ncid, id, &dim), "read a dimension");
        check(nc_inq_dimlen(ncid, dim, &len), "read a dimension length");
        if (len != gather.num_global() && error.empty()) {
          error = "Restart '" + path + "': '" + name + "' has " +
                  std::to_string(len) + " cells; this grid has " +
                  std::to_string(gather.num_global()) + ".";
        }
        array_names.push_back(name);
      } else if (ndims == 0) {
        long long value = 0;
        check(nc_get_var_longlong(ncid, id, &value), "read " +
                                                         std::string(name));
        int_names.push_back(name);
        int_values.push_back(value);
      }
    }
  }
  share_error(error, comm, 0);

  auto share_names = [&](std::vector<std::string> &names) {
    int count = static_cast<int>(names.size());
    MPI_Bcast(&count, 1, MPI_INT, 0, comm);
    names.resize(static_cast<std::size_t>(count));
    for (auto &name : names) {
      int length = static_cast<int>(name.size());
      MPI_Bcast(&length, 1, MPI_INT, 0, comm);
      name.resize(static_cast<std::size_t>(length));
      MPI_Bcast(name.data(), length, MPI_CHAR, 0, comm);
    }
  };
  share_names(array_names);
  share_names(int_names);
  int_values.resize(int_names.size());
  MPI_Bcast(int_values.data(), static_cast<int>(int_values.size()),
            MPI_LONG_LONG, 0, comm);

  MemoryRestartStore store;
  for (std::size_t k = 0; k < int_names.size(); ++k) {
    store.write_int(int_names[k], int_values[k]);
  }
  std::vector<double> global(root ? gather.num_global() : 0);
  std::vector<double> mine(gather.num_local());
  for (const auto &name : array_names) {
    if (root && error.empty()) {
      int id = -1;
      if (check(nc_inq_varid(ncid, name.c_str(), &id), "find " + name)) {
        check(nc_get_var_double(ncid, id, global.data()), "read " + name);
      }
    }
    gather.scatter(global, mine);
    store.write_array(name, mine);
  }
  if (root && ncid >= 0) {
    nc_close(ncid);
  }
  share_error(error, comm, 0);
  return store;
}

#else // !EMULATOR_HAVE_NETCDF

bool have_restart_files() { return false; }

void write_restart_file(const std::string &path, const MemoryRestartStore &,
                        const grid::GlobalGather &, MPI_Comm,
                        const std::map<std::string, std::string> &) {
  throw std::runtime_error("Restart '" + path + "': this build has no netCDF.");
}

MemoryRestartStore read_restart_file(const std::string &path,
                                     const grid::GlobalGather &, MPI_Comm) {
  throw std::runtime_error("Restart '" + path + "': this build has no netCDF.");
}

#endif

} // namespace coupling
} // namespace emulator
