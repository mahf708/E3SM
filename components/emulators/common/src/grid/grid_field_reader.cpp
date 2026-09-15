/**
 * @file grid_field_reader.cpp
 * @brief read_grid_fields over serial netCDF-C.
 */

#include "grid_field_reader.hpp"

#include <stdexcept>

#ifdef EMULATOR_HAVE_NETCDF

#include <netcdf.h>

#include <cmath>

namespace emulator {
namespace grid {

std::vector<GridField> read_grid_fields(const std::string &path,
                                        const std::vector<std::string> &names,
                                        int ny, int nx) {
  auto check = [&](int status, const std::string &what) {
    if (status != NC_NOERR) {
      throw std::runtime_error("Grid field file '" + path + "': cannot " +
                               what + ": " + nc_strerror(status));
    }
  };
  int ncid = -1;
  check(nc_open(path.c_str(), NC_NOWRITE, &ncid), "open");
  struct Closer {
    int id;
    ~Closer() { nc_close(id); }
  } closer{ncid};

  const auto n = static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
  std::vector<GridField> out;
  out.reserve(names.size());
  for (const auto &name : names) {
    int varid = 0;
    if (nc_inq_varid(ncid, name.c_str(), &varid) != NC_NOERR) {
      throw std::runtime_error("Grid field file '" + path + "' has no variable '" +
                               name + "'.");
    }
    int ndims = 0;
    check(nc_inq_varndims(ncid, varid, &ndims), "read the rank of " + name);
    std::vector<int> dimids(static_cast<std::size_t>(ndims));
    check(nc_inq_vardimid(ncid, varid, dimids.data()),
          "read the dimensions of " + name);
    std::vector<std::size_t> lens;
    for (const int d : dimids) {
      std::size_t len = 0;
      check(nc_inq_dimlen(ncid, d, &len), "read a dimension of " + name);
      lens.push_back(len);
    }
    const bool plain = lens.size() == 2;
    const bool one_time = lens.size() == 3 && lens[0] == 1;
    const auto shape_ok =
        (plain || one_time) && lens[lens.size() - 2] == static_cast<std::size_t>(ny) &&
        lens.back() == static_cast<std::size_t>(nx);
    if (!shape_ok) {
      std::string shape;
      for (const auto l : lens) {
        shape += (shape.empty() ? "" : " x ") + std::to_string(l);
      }
      throw std::runtime_error(
          "Grid field file '" + path + "': '" + name + "' is " + shape +
          "; expected " + std::to_string(ny) + " x " + std::to_string(nx) +
          " (lat x lon), optionally with a leading time of 1.");
    }

    GridField field;
    field.name = name;
    field.values.resize(n);
    check(nc_get_var_double(ncid, varid, field.values.data()), "read " + name);

    double fill = 0.0;
    const bool have_fill =
        nc_get_att_double(ncid, varid, "_FillValue", &fill) == NC_NOERR &&
        std::isfinite(fill);
    for (const double v : field.values) {
      if (!std::isfinite(v)) {
        ++field.non_finite;
      } else if (std::abs(v) >= 1e30 || (have_fill && v == fill)) {
        ++field.fill_like;
      }
    }
    out.push_back(std::move(field));
  }
  return out;
}

} // namespace grid
} // namespace emulator

#else

namespace emulator {
namespace grid {

std::vector<GridField> read_grid_fields(const std::string &path,
                                        const std::vector<std::string> &,
                                        int, int) {
  throw std::runtime_error("Cannot read '" + path +
                           "': this build has no netCDF. Reconfigure with "
                           "-DEMULATOR_ENABLE_NETCDF=ON.");
}

} // namespace grid
} // namespace emulator

#endif
