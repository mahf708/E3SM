/**
 * @file emulator_component.cpp
 * @brief Implementation of EmulatorComponent.
 */

#include "emulator_component.hpp"

#include "create_inference_backend.hpp"
#include "grid_field_reader.hpp"
#include "restart_file.hpp"
#include "scrip_reader.hpp"

#include <mpi.h>

#include <filesystem>
#include <stdexcept>

namespace emulator {

namespace {

std::string resolve(const std::string &path, const std::string &base_dir) {
  if (path.empty() || path.front() == '/' || base_dir.empty()) {
    return path;
  }
  return base_dir + "/" + path;
}

} // namespace

EmulatorComponent::EmulatorComponent(EmulatorType type, std::string name,
                                     coupling::Exchange &exchange)
    : Emulator(type, -1, name), m_exchange(exchange) {}

EmulatorComponent::~EmulatorComponent() = default;

void EmulatorComponent::create_instance(int comm, int comp_id,
                                        const std::string &input_file,
                                        const std::string &log_file,
                                        int run_type, int start_ymd,
                                        int start_tod) {
  (void)log_file;
  m_run_type = run_type;
  m_comm = comm;
  m_id = comp_id;
  set_start_time({start_ymd, start_tod});
  if (input_file.empty()) {
    return; // unconfigured: a caller may still set_grid_data()
  }
  if (!std::filesystem::exists(input_file)) {
    throw std::invalid_argument(m_name + ": the input file '" + input_file +
                                "' does not exist.");
  }
  m_input = std::make_unique<config::Section>(
      config::Section::load_file(input_file));
  const auto &in = *m_input;
  in.only({"spec", "coupler_dt", "grid", "initial_condition", "inference"});
  const auto parent = std::filesystem::path(input_file).parent_path().string();
  m_base_dir = parent;
  m_spec = std::make_unique<model::ModelSpec>(model::ModelSpec::read(
      config::Section::load_spec(resolve(in.string("spec"), m_base_dir))));
  m_coupler_dt = static_cast<int>(in.integer("coupler_dt"));
  setup_grid(in.section("grid"), m_base_dir);
}

void EmulatorComponent::setup_grid(const config::Section &g,
                                   const std::string &base_dir) {
  g.only({"file", "domain", "mask_variable", "publish_as", "shared_from"});
  MPI_Comm c_comm = MPI_Comm_f2c(m_comm);
  int rank = 0, size = 1;
  MPI_Comm_rank(c_comm, &rank);
  MPI_Comm_size(c_comm, &size);
  const auto domain = g.string("domain");

  if (domain == "shared") {
    // Collective, so a rank without the other component fails with the
    // others instead of leaving them waiting.
    const auto from = g.string("shared_from");
    const bool here = coupling::has_domain(m_exchange, from);
    coupling::SharedDomain shared;
    long local[2] = {here ? 1 : 0, 0};
    if (here) {
      shared = coupling::shared_domain(m_exchange, from);
      local[1] = static_cast<long>(shared.domain.size());
    }
    long least = 0, cells = 0;
    MPI_Allreduce(&local[0], &least, 1, MPI_LONG, MPI_MIN, c_comm);
    MPI_Allreduce(&local[1], &cells, 1, MPI_LONG, MPI_SUM, c_comm);
    if (least == 0) {
      throw std::runtime_error(
          m_name + ": no domain from '" + from + "' on at least one of this "
          "component's ranks. It takes that component's domain, so needs it in "
          "this run, created first, on the same ranks (NTASKS and ROOTPE equal "
          "to its).");
    }
    if (static_cast<std::size_t>(cells) != shared.num_global) {
      throw std::runtime_error(
          m_name + ": this component's ranks hold " + std::to_string(cells) +
          " of " + from + "'s " + std::to_string(shared.num_global) +
          " cells. It must run on exactly that component's ranks.");
    }
    m_decomp = grid::Decomposition::contiguous_blocks(shared.num_global, size,
                                                      rank);
    if (m_decomp.global_ids() != shared.domain.global_ids) {
      throw std::runtime_error(m_name + ": " + from + "'s decomposition is not "
                               "the contiguous blocks this component uses.");
    }
    set_domain(std::move(shared.domain), shared.nx, shared.ny,
               shared.num_global);
    return;
  }

  m_grid = grid::read_scrip(resolve(g.string("file"), base_dir));
  m_have_grid = true;
  m_decomp = grid::Decomposition::contiguous_blocks(m_grid.size(), size, rank);
  grid::Domain d;
  if (domain == "full") {
    d = grid::Domain::full(m_grid, m_decomp);
  } else if (domain == "ocean_mask") {
    const auto mask = grid::read_grid_fields(
        resolve(m_input->string("initial_condition"), base_dir),
        {g.string("mask_variable")}, m_grid.ny, m_grid.nx);
    d = grid::Domain::masked(m_grid, m_decomp, mask.at(0).values);
  } else {
    throw std::invalid_argument(g.where() + ".domain: '" + domain +
                                "' is not full, ocean_mask or shared.");
  }
  if (g.has("publish_as")) {
    coupling::publish_domain(m_exchange, g.string("publish_as"),
                             {d, m_grid.nx, m_grid.ny, m_grid.size()});
  }
  set_domain(std::move(d), m_grid.nx, m_grid.ny, m_grid.size());
}

EmulatorComponent::CouplingFields EmulatorComponent::coupling_fields() const {
  CouplingFields f;
  if (m_spec) {
    f.imports = m_spec->imports;
    f.exports = m_spec->exports;
  }
  return f;
}

void EmulatorComponent::init_impl() {
  if (!m_spec) {
    return;
  }
  if (start_time().ymd < 0) {
    throw std::invalid_argument(m_name + ": no start time from the driver.");
  }
  MPI_Comm comm = MPI_Comm_f2c(m_comm);
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  std::shared_ptr<inference::InferenceBackend> backend;
  if (m_spec->layout && rank == 0) {
    const auto inf = m_input->section("inference");
    inference::InferenceConfig ic;
    ic.backend = inf.string_or("backend", "libtorch");
    ic.model_path = resolve(inf.string("model_path"), m_base_dir);
    ic.set("device", "cuda");
    for (const auto &key : inf.keys()) {
      if (key != "backend" && key != "model_path") {
        ic.set(key, inf.string(key));
      }
    }
    // The component's ranks and communicator, and the whole grid: the
    // network runs on this rank alone, on every column gathered here.
    auto context = inference::make_context(m_comm);
    context.gathered = true;
    const auto &g = m_grid;
    std::vector<int> all(g.size());
    for (std::size_t k = 0; k < all.size(); ++k) {
      all[k] = static_cast<int>(k + 1);
    }
    context.set_grid(g.nx, g.ny, static_cast<int>(g.size()), all.data(),
                     g.lat.data(), g.lon.data(), static_cast<int>(g.size()));
    backend = inference::create_backend(ic, context);
  }

  auto geometry = m_have_grid
                      ? model::Geometry::from_grid(comm, m_grid, m_decomp,
                                                   domain().mask)
                      : model::Geometry::from_domain(comm, domain(), m_decomp);
  m_model = std::make_unique<model::EmulatedModel>(
      *m_spec, m_coupler_dt, std::move(geometry), backend, &m_exchange);
  const auto names = m_model->initial_condition_names();
  std::vector<grid::GridField> initial;
  if (!names.empty()) {
    initial = grid::read_grid_fields(
        resolve(m_input->string("initial_condition"), m_base_dir), names,
        m_grid.ny, m_grid.nx);
  }
  if (m_restart_file.empty()) {
    if (m_run_type != 0) {
      throw std::invalid_argument(
          m_name + ": a continue or branch run needs a restart file "
          "(set_restart_file); starting from the initial condition would "
          "silently begin a different run.");
    }
    m_model->initialize(start_time(), initial);
  } else {
    const auto &g = m_model->geometry();
    auto store = coupling::read_restart_file(m_restart_file, *g.gather, comm);
    m_model->restart(store, initial);
  }
  if (is_coupled()) {
    m_model->initial_exports(start_time(), imports(), mutable_exports());
  }
}

void EmulatorComponent::run_impl(int dt) {
  if (!m_model) {
    return;
  }
  const auto now = current_time();
  if (now.ymd < 0) {
    throw std::logic_error(
        m_name + ": run without a model time. An emulated component needs the "
        "driver's time each step (emulator_run_at).");
  }
  if (dt != m_coupler_dt) {
    throw std::invalid_argument(m_name + ": the driver's step is " +
                                std::to_string(dt) + " s but coupler_dt is " +
                                std::to_string(m_coupler_dt) + " s.");
  }
  m_model->run(now, imports(), mutable_exports());
}

void EmulatorComponent::set_restart_file(const std::string &path) {
  if (is_initialized()) {
    throw std::logic_error(m_name + ": set_restart_file after initialize.");
  }
  m_restart_file = path;
}

void EmulatorComponent::write_restart(const std::string &path) const {
  if (!m_model) {
    return;
  }
  coupling::MemoryRestartStore store;
  m_model->save_to(store);
  const auto &g = m_model->geometry();
  const auto &clock = m_model->clock();
  write_restart_file(path, store, *g.gather, g.comm,
                     {{"component", m_name},
                      {"spec", m_spec->name},
                      {"clock", clock.to_string()}});
}

void EmulatorComponent::final_impl() { m_model.reset(); }

} // namespace emulator
