/**
 * @file scrip_reader.cpp
 * @brief SCRIP grid reader, over serial netCDF-C.
 */

#include "scrip_reader.hpp"

#include <stdexcept>

#ifdef EMULATOR_HAVE_NETCDF

#include <netcdf.h>

#include <algorithm>
#include <cctype>
#include <numbers>
#include <vector>

namespace emulator {
namespace grid {

namespace {

/// Owns an open netCDF id; every error says which file and what was asked.
class NcFile {
public:
  explicit NcFile(std::string path) : m_path(std::move(path)) {
    check(nc_open(m_path.c_str(), NC_NOWRITE, &m_ncid), "open");
  }
  ~NcFile() { nc_close(m_ncid); }
  NcFile(const NcFile &) = delete;
  NcFile &operator=(const NcFile &) = delete;

  void check(int status, const std::string &what) const {
    if (status != NC_NOERR) {
      throw std::runtime_error("SCRIP file '" + m_path + "': cannot " + what +
                               ": " + nc_strerror(status));
    }
  }

  bool has_var(const char *name) const {
    int varid = 0;
    return nc_inq_varid(m_ncid, name, &varid) == NC_NOERR;
  }

  int varid(const char *name) const {
    int id = 0;
    check(nc_inq_varid(m_ncid, name, &id),
          std::string("find variable '") + name + "'");
    return id;
  }

  std::size_t var_length(const char *name) const {
    const int id = varid(name);
    int ndims = 0;
    check(nc_inq_varndims(m_ncid, id, &ndims), "read the rank of " +
                                                   std::string(name));
    std::vector<int> dims(static_cast<std::size_t>(ndims));
    check(nc_inq_vardimid(m_ncid, id, dims.data()),
          "read the dimensions of " + std::string(name));
    std::size_t n = 1;
    for (const int d : dims) {
      std::size_t len = 0;
      check(nc_inq_dimlen(m_ncid, d, &len),
            "read a dimension of " + std::string(name));
      n *= len;
    }
    return n;
  }

  std::string text_attribute(const char *var, const char *att) const {
    const int id = varid(var);
    std::size_t len = 0;
    if (nc_inq_attlen(m_ncid, id, att, &len) != NC_NOERR) {
      return {};
    }
    std::string value(len, '\0');
    check(nc_get_att_text(m_ncid, id, att, value.data()),
          "read " + std::string(var) + ":" + att);
    value.erase(std::find(value.begin(), value.end(), '\0'), value.end());
    return value;
  }

  std::vector<double> doubles(const char *name) const {
    std::vector<double> out(var_length(name));
    check(nc_get_var_double(m_ncid, varid(name), out.data()),
          "read " + std::string(name));
    return out;
  }

  std::vector<int> ints(const char *name) const {
    std::vector<int> out(var_length(name));
    check(nc_get_var_int(m_ncid, varid(name), out.data()),
          "read " + std::string(name));
    return out;
  }

  const std::string &path() const { return m_path; }

private:
  std::string m_path;
  int m_ncid = -1;
};

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

/// Degrees per unit of this coordinate, from its units attribute.
double to_degrees(const NcFile &file, const char *var) {
  const std::string units = lower(file.text_attribute(var, "units"));
  if (units.rfind("degree", 0) == 0) {
    return 1.0;
  }
  if (units.rfind("radian", 0) == 0) {
    return 180.0 / std::numbers::pi;
  }
  throw std::runtime_error(
      "SCRIP file '" + file.path() + "': " + var + " has units '" + units +
      "'. Expected degrees or radians; refusing to guess, because a guess "
      "that is wrong still produces a plausible-looking grid.");
}

} // namespace

bool have_scrip_reader() { return true; }

HorizontalGrid read_scrip(const std::string &path, bool expect_global) {
  NcFile file(path);

  HorizontalGrid grid;
  grid.name = path;

  const double lat_scale = to_degrees(file, "grid_center_lat");
  const double lon_scale = to_degrees(file, "grid_center_lon");
  grid.lat = file.doubles("grid_center_lat");
  grid.lon = file.doubles("grid_center_lon");
  for (auto &v : grid.lat) {
    v *= lat_scale;
  }
  for (auto &v : grid.lon) {
    v *= lon_scale;
  }

  grid.area = file.doubles("grid_area");
  const std::string area_units = lower(file.text_attribute("grid_area", "units"));
  if (area_units.find("deg") != std::string::npos) {
    throw std::runtime_error(
        "SCRIP file '" + path + "': grid_area has units '" + area_units +
        "'. The coupler's areas are solid angles in radians^2.");
  }

  if (file.has_var("grid_imask")) {
    grid.imask = file.ints("grid_imask");
  } else {
    grid.imask.assign(grid.lat.size(), 1);
  }

  const auto dims = file.ints("grid_dims");
  if (dims.size() == 2) {
    grid.nx = dims[0];
    grid.ny = dims[1];
  } else if (dims.size() == 1) {
    grid.nx = dims[0];
    grid.ny = 1;
  } else {
    throw std::runtime_error("SCRIP file '" + path + "': grid_dims has " +
                             std::to_string(dims.size()) +
                             " entries; expected 1 or 2.");
  }

  grid.validate(expect_global);
  return grid;
}

} // namespace grid
} // namespace emulator

#else // EMULATOR_HAVE_NETCDF

namespace emulator {
namespace grid {

bool have_scrip_reader() { return false; }

HorizontalGrid read_scrip(const std::string &path, bool) {
  throw std::runtime_error(
      "Cannot read the SCRIP grid '" + path +
      "': this build has no netCDF. Reconfigure with "
      "-DEMULATOR_ENABLE_NETCDF=ON.");
}

} // namespace grid
} // namespace emulator

#endif // EMULATOR_HAVE_NETCDF
