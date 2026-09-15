#include "ace_operators.hpp"
#include "calendar.hpp"
#include "emulator_component.hpp"
#include "grid_field_reader.hpp"
#include "yaml_config.hpp"

#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
using emulator::config::Section;
using emulator::coupling::ModelTime;

std::string resolve(const std::string &value, const std::filesystem::path &base) {
  return (base / value).lexically_normal().string();
}

int positive(const Section &s, const std::string &key) {
  const auto n = s.integer(key);
  if (n <= 0 || n > std::numeric_limits<int>::max())
    throw std::invalid_argument(key + " must be a positive int");
  return static_cast<int>(n);
}

ModelTime advance(ModelTime time, int dt) {
  const int days[] = {31,28,31,30,31,30,31,31,30,31,30,31};
  long long seconds = static_cast<long long>(time.tod) + dt;
  int y = time.ymd / 10000, m = time.ymd / 100 % 100, d = time.ymd % 100;
  while (seconds >= 86400) {
    seconds -= 86400;
    if (++d > days[m - 1]) { d = 1; if (++m > 12) { m = 1; ++y; } }
  }
  return {y * 10000 + m * 100 + d, static_cast<int>(seconds)};
}

std::string joined(const std::vector<emulator::fields::FieldSpec> &fields) {
  std::string list;
  for (const auto &f : fields) list += (list.empty() ? "" : ":") + f.name;
  return list;
}

void run(const std::string &path, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  const auto base = std::filesystem::absolute(path).parent_path();
  const auto cfg = Section::load_file(path);
  cfg.only({"component", "start_ymd", "start_tod", "steps", "surface",
            "output", "restart_in", "restart_out"});
  ModelTime now{positive(cfg, "start_ymd"), static_cast<int>(cfg.integer("start_tod"))};
  emulator::coupling::julian_day_noleap(now.ymd, now.tod);
  const int steps = positive(cfg, "steps");
  const auto input_path = resolve(cfg.string("component"), base);
  const auto input = Section::load_file(input_path);
  const int dt = positive(input, "coupler_dt");
  const auto spec = emulator::model::ModelSpec::read(Section::load_spec(
      resolve(input.string("spec"), std::filesystem::path(input_path).parent_path())));

  emulator::atm::register_atm_operators();
  emulator::coupling::Exchange exchange;
  emulator::EmulatorComponent atm(emulator::EmulatorType::ATM_COMP, "emulatoratm", exchange);
  atm.create_instance(MPI_Comm_c2f(comm), 1, input_path, "", cfg.has("restart_in") ? 1 : 0,
                      now.ymd, now.tod);
  const auto n = static_cast<std::size_t>(atm.get_num_local_cols());
  std::vector<double> imports(n * spec.imports.size()), exports(n * spec.exports.size());
  const auto surface = cfg.section("surface");
  surface.only({"file", "variables"});
  const auto variables = surface.section("variables");
  std::vector<std::string> names;
  for (const auto &field : spec.imports) names.push_back(variables.string(field.name));
  if (variables.keys().size() != names.size())
    throw std::invalid_argument("surface.variables must map exactly the declared imports");
  if (!names.empty()) {
    const auto values = emulator::grid::read_grid_fields(resolve(surface.string("file"), base),
                                                         names, atm.get_ny(), atm.get_nx());
    for (std::size_t f = 0; f < names.size(); ++f) {
      if (values[f].unusable()) throw std::runtime_error("invalid prescribed surface field: " + names[f]);
      for (std::size_t p = 0; p < n; ++p)
        imports[p * names.size() + f] = values[f].values[atm.domain().global_ids[p] - 1];
    }
  }
  atm.set_coupler_field_lists(joined(spec.imports), joined(spec.exports));
  EmulatorCouplingDesc buffers{imports.data(), exports.data(), static_cast<int>(spec.imports.size()),
                              static_cast<int>(spec.exports.size()), static_cast<int>(n)};
  atm.setup_coupling(buffers);
  if (cfg.has("restart_in")) atm.set_restart_file(resolve(cfg.string("restart_in"), base));
  std::ofstream output;
  if (rank == 0) {
    output.open(resolve(cfg.string("output"), base));
    if (!output) throw std::runtime_error("cannot open output CSV");
    output << "ymd,tod";
    for (const auto &field : spec.exports) output << ',' << field.name;
    output << '\n' << std::setprecision(17);
  }
  std::vector<double> areas(n);
  atm.get_cols_area(areas.data());
  const double local_area = std::accumulate(areas.begin(), areas.end(), 0.0);
  double total_area;
  MPI_Allreduce(&local_area, &total_area, 1, MPI_DOUBLE, MPI_SUM, comm);
  auto record = [&] {
    if (rank == 0) output << now.ymd << ',' << now.tod;
    for (std::size_t f = 0; f < spec.exports.size(); ++f) {
      double local = 0, total = 0;
      for (std::size_t p = 0; p < n; ++p) local += areas[p] * exports[p * spec.exports.size() + f];
      MPI_Reduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
      if (rank == 0) output << ',' << total / total_area;
    }
    if (rank == 0) output << '\n';
  };
  atm.initialize();
  record();
  for (int step = 0; step < steps; ++step) {
    now = advance(now, dt);
    atm.run(dt, now);
    record();
  }
  if (cfg.has("restart_out")) atm.write_restart(resolve(cfg.string("restart_out"), base));
  atm.finalize();
  if (rank == 0) {
    output.flush();
    if (!output) throw std::runtime_error("failed writing output CSV");
    std::cout << "emulatoratm completed " << steps << " steps at " << now.to_string() << '\n';
  }
}
} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  try {
    if (argc != 2) throw std::invalid_argument("usage: emulatoratm_driver run.yaml");
    run(argv[1], comm);
  } catch (const std::exception &e) {
    std::cerr << "emulatoratm: " << e.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
  MPI_Comm_free(&comm);
  MPI_Finalize();
  return 0;
}
