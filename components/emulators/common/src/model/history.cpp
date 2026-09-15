/**
 * @file history.cpp
 * @brief History output over serial netCDF-C, written on the root.
 */

#include "history.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

#ifdef EMULATOR_HAVE_NETCDF
#include <netcdf.h>
#endif

namespace emulator {
namespace model {

namespace {

constexpr double kFill = 1.0e20;

std::string stamp(coupling::ModelTime t) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d-%05d", t.ymd / 10000,
                (t.ymd / 100) % 100, t.ymd % 100, t.tod);
  return buf;
}

} // namespace

History::History(const config::Section &o, const Geometry &geometry, int nx,
                 int ny, int coupler_dt, std::string component)
    : m_geometry(&geometry), m_nx(nx), m_ny(ny), m_coupler_dt(coupler_dt),
      m_component(std::move(component)) {
  o.only({"prefix", "interval", "fields"});
  m_prefix = o.string_or("prefix", m_component + ".h");
  m_interval = o.string_or("interval", "monthly");
  if (m_interval != "monthly") {
    const char unit = m_interval.empty() ? '\0' : m_interval.back();
    const std::string count = m_interval.substr(0, m_interval.size() - 1);
    const bool digits = !count.empty() &&
                        count.find_first_not_of("0123456789") == std::string::npos;
    if (!digits || (unit != 'd' && unit != 'h') || std::stol(count) <= 0) {
      throw std::invalid_argument(o.where() + ".interval: '" + m_interval +
                                  "' is not monthly, <n>d or <n>h.");
    }
    const long seconds = std::stol(count) * (unit == 'd' ? 86400 : 3600);
    if (seconds % coupler_dt != 0) {
      throw std::invalid_argument(
          o.where() + ".interval: " + m_interval + " is not a whole number of " +
          std::to_string(coupler_dt) + " s coupler steps.");
    }
    m_interval_steps = static_cast<int>(seconds / coupler_dt);
  }
  const auto texts = o.names("fields");
  if (texts.empty()) {
    throw std::invalid_argument(o.where() + ".fields: no fields to write.");
  }
  std::map<std::string, int> uses;
  for (const auto &text : texts) {
    m_fields.push_back(FieldRef::parse(text, o.where() + ".fields"));
    ++uses[m_fields.back().name()];
  }
  for (const auto &r : m_fields) {
    // A name two sets share is qualified by its set: state_TS, upper_TS.
    std::string name = r.name();
    if (uses[name] > 1) {
      name = r.to_string();
      name[name.find('.')] = '_';
    }
    m_names.push_back(name);
  }
  m_sums.assign(m_fields.size(),
                std::vector<double>(m_geometry->num_local(), 0.0));
}

bool History::closes_interval(coupling::ModelTime now) const {
  if (m_interval_steps > 0) {
    return m_steps % m_interval_steps == 0;
  }
  return now.tod == 0 && now.ymd % 100 == 1;
}

void History::after_step(coupling::ModelTime now, const Fields &f) {
  if (now == m_last) {
    return; // a repeated driver call
  }
  if (m_samples == 0) {
    m_start = m_last; // unset (-1) for the run's first interval
  }
  m_last = now;
  ++m_steps;
  for (std::size_t k = 0; k < m_fields.size(); ++k) {
    const auto values = m_fields[k].read(f);
    if (values.size() != m_sums[k].size()) {
      throw std::runtime_error("history: '" + m_fields[k].to_string() +
                               "' has " + std::to_string(values.size()) +
                               " values for " +
                               std::to_string(m_sums[k].size()) + " cells.");
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
      m_sums[k][i] += values[i];
    }
  }
  ++m_samples;
  if (closes_interval(now)) {
    write(now);
    for (auto &s : m_sums) {
      std::fill(s.begin(), s.end(), 0.0);
    }
    m_samples = 0;
  }
}

void History::save_to(coupling::RestartStore &store) const {
  store.write_int("history.samples", m_samples);
  store.write_int("history.steps", m_steps);
  store.write_int("history.start_ymd", m_start.ymd);
  store.write_int("history.start_tod", m_start.tod);
  store.write_int("history.last_ymd", m_last.ymd);
  store.write_int("history.last_tod", m_last.tod);
  for (std::size_t k = 0; k < m_fields.size(); ++k) {
    store.write_array("history.sum." + m_fields[k].to_string(), m_sums[k]);
  }
}

bool History::load_from(coupling::RestartStore &store) {
  std::int64_t samples = 0, steps = 0, v[4] = {0, 0, 0, 0};
  bool ok = store.read_int("history.samples", samples) &&
            store.read_int("history.steps", steps) &&
            store.read_int("history.start_ymd", v[0]) &&
            store.read_int("history.start_tod", v[1]) &&
            store.read_int("history.last_ymd", v[2]) &&
            store.read_int("history.last_tod", v[3]);
  for (std::size_t k = 0; ok && k < m_fields.size(); ++k) {
    ok = store.read_array("history.sum." + m_fields[k].to_string(), m_sums[k]);
  }
  if (!ok) {
    for (auto &s : m_sums) {
      std::fill(s.begin(), s.end(), 0.0);
    }
    return false;
  }
  m_samples = samples;
  m_steps = steps;
  m_start = {static_cast<int>(v[0]), static_cast<int>(v[1])};
  m_last = {static_cast<int>(v[2]), static_cast<int>(v[3])};
  return true;
}

void History::write(coupling::ModelTime end) {
  const auto &g = *m_geometry;
  const auto &gather = *g.gather;
  const bool root = gather.is_root();
  const std::size_t n = gather.num_global();
  const auto path = m_prefix + "." + stamp(end) + ".nc";

  std::vector<double> lat(root ? n : 0), lon(root ? n : 0), mask(root ? n : 0);
  gather.gather(g.lat, lat);
  gather.gather(g.lon, lon);
  gather.gather(g.domain_mask, mask);

#ifdef EMULATOR_HAVE_NETCDF
  std::string error;
  int ncid = -1;
  std::vector<int> ids;
  const auto nx = static_cast<std::size_t>(m_nx);
  const auto ny = static_cast<std::size_t>(m_ny);
  auto check = [&](int status, const std::string &what) {
    if (status != NC_NOERR && error.empty()) {
      error = "history '" + path + "': cannot " + what + ": " +
              nc_strerror(status);
    }
    return status == NC_NOERR;
  };
  if (root) {
    if (nx * ny != n) {
      error = "history '" + path + "': " + std::to_string(n) +
              " cells are not a " + std::to_string(ny) + " x " +
              std::to_string(nx) + " grid.";
    }
    if (error.empty() &&
        check(nc_create(path.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid),
              "create")) {
      int dlat = -1, dlon = -1, vlat = -1, vlon = -1;
      check(nc_def_dim(ncid, "lat", ny, &dlat), "define lat");
      check(nc_def_dim(ncid, "lon", nx, &dlon), "define lon");
      check(nc_def_var(ncid, "lat", NC_DOUBLE, 1, &dlat, &vlat), "define lat");
      check(nc_def_var(ncid, "lon", NC_DOUBLE, 1, &dlon, &vlon), "define lon");
      check(nc_put_att_text(ncid, vlat, "units", 13, "degrees_north"),
            "write units");
      check(nc_put_att_text(ncid, vlon, "units", 12, "degrees_east"),
            "write units");
      const auto text = [&](const char *key, const std::string &value) {
        check(nc_put_att_text(ncid, NC_GLOBAL, key, value.size(),
                              value.c_str()),
              std::string("write ") + key);
      };
      text("component", m_component);
      text("interval", m_interval);
      text("start", m_start.ymd < 0 ? "run start" : stamp(m_start));
      text("end", stamp(end));
      long long samples = m_samples;
      check(nc_put_att_longlong(ncid, NC_GLOBAL, "samples", NC_INT64, 1,
                                &samples),
            "write samples");
      int dims[2] = {dlat, dlon};
      for (std::size_t k = 0; k < m_fields.size(); ++k) {
        int id = -1;
        check(nc_def_var(ncid, m_names[k].c_str(), NC_DOUBLE, 2, dims, &id),
              "define " + m_names[k]);
        check(nc_put_att_double(ncid, id, "_FillValue", NC_DOUBLE, 1, &kFill),
              "write _FillValue");
        const auto source = m_fields[k].to_string();
        check(nc_put_att_text(ncid, id, "source", source.size(),
                              source.c_str()),
              "write source");
        check(nc_put_att_text(ncid, id, "cell_methods", 10, "time: mean"),
              "write cell_methods");
        ids.push_back(id);
      }
      check(nc_enddef(ncid), "leave define mode");
      std::vector<double> rows(ny), cols(nx);
      for (std::size_t j = 0; j < ny && error.empty(); ++j) {
        rows[j] = lat[j * nx];
      }
      for (std::size_t i = 0; i < nx && error.empty(); ++i) {
        cols[i] = lon[i];
      }
      if (error.empty()) {
        check(nc_put_var_double(ncid, vlat, rows.data()), "write lat");
        check(nc_put_var_double(ncid, vlon, cols.data()), "write lon");
      }
    }
  }
  std::vector<double> global(root ? n : 0);
  for (std::size_t k = 0; k < m_fields.size(); ++k) {
    gather.gather(m_sums[k], global);
    if (root && error.empty()) {
      for (std::size_t c = 0; c < n; ++c) {
        global[c] = mask[c] == 0.0 ? kFill
                                   : global[c] / static_cast<double>(m_samples);
      }
      check(nc_put_var_double(ncid, ids[k], global.data()),
            "write " + m_names[k]);
    }
  }
  if (root && ncid >= 0) {
    check(nc_close(ncid), "close");
  }
  int length = static_cast<int>(error.size());
  MPI_Bcast(&length, 1, MPI_INT, 0, g.comm);
  error.resize(static_cast<std::size_t>(length));
  if (length > 0) {
    MPI_Bcast(error.data(), length, MPI_CHAR, 0, g.comm);
    throw std::runtime_error(error);
  }
  m_written.push_back(path);
#else
  throw std::runtime_error("history '" + path + "': this build has no netCDF.");
#endif
}

} // namespace model
} // namespace emulator
