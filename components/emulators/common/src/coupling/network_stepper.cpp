/**
 * @file network_stepper.cpp
 * @brief Implementation of NetworkStepper.
 */

#include "network_stepper.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace emulator {
namespace coupling {

NetworkStepper::NetworkStepper(
    fields::ChannelLayout layout, MPI_Comm comm,
    const grid::GlobalGather &gather, int nx, int ny,
    std::shared_ptr<inference::InferenceBackend> backend)
    : m_layout(std::move(layout)), m_comm(comm), m_gather(&gather), m_nx(nx),
      m_ny(ny), m_backend(std::move(backend)), m_inputs(gather.num_local()),
      m_prediction(gather.num_local()) {
  m_layout.validate();
  if (nx <= 0 || ny <= 0 ||
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) !=
          gather.num_global()) {
    throw std::invalid_argument(
        "Network '" + m_layout.name + "': an " + std::to_string(ny) + " x " +
        std::to_string(nx) + " tensor for a grid of " +
        std::to_string(gather.num_global()) + " cells.");
  }
  if (gather.is_root() && !m_backend) {
    throw std::invalid_argument("Network '" + m_layout.name +
                                "': the root rank needs a backend.");
  }
  for (const auto &name : m_layout.inputs) {
    m_inputs.add(name);
  }
  for (const auto &name : m_layout.outputs) {
    m_prediction.add(name);
  }
  if (gather.is_root()) {
    m_global_in.resize(m_layout.inputs.size() * gather.num_global());
    m_global_out.resize(m_layout.outputs.size() * gather.num_global());
  }
}

void NetworkStepper::step(std::int64_t step_index) {
  const std::size_t ncells = m_gather->num_global();
  const bool root = m_gather->is_root();
  const std::span<double> no_global;

  // 1. gather
  for (std::size_t c = 0; c < m_layout.inputs.size(); ++c) {
    const auto dest = root ? std::span<double>(m_global_in).subspan(
                                 c * ncells, ncells)
                           : no_global;
    m_gather->gather(m_inputs.get(m_layout.inputs[c]), dest);
  }

  // 2. infer, and 3. check, on the root; the verdict goes to everyone.
  std::string failure;
  if (root) {
    const auto cin = static_cast<std::int64_t>(m_layout.inputs.size());
    const auto cout = static_cast<std::int64_t>(m_layout.outputs.size());
    inference::TensorMap in;
    in.wrap("inputs", std::as_const(m_global_in).data(), {1, cin, m_ny, m_nx});
    inference::TensorMap out;
    out.wrap("outputs", m_global_out.data(), {1, cout, m_ny, m_nx});
    try {
      m_backend->set_step(step_index);
      m_backend->infer(in, out);
      for (std::size_t c = 0; c < m_layout.outputs.size() && failure.empty();
           ++c) {
        std::size_t bad = 0;
        std::size_t first = 0;
        for (std::size_t k = 0; k < ncells; ++k) {
          if (!std::isfinite(m_global_out[c * ncells + k])) {
            if (bad++ == 0) {
              first = k;
            }
          }
        }
        if (bad > 0) {
          failure = "output '" + m_layout.outputs[c] + "' has " +
                    std::to_string(bad) + " non-finite values (first at cell " +
                    std::to_string(first) +
                    "). One NaN in a global network's state spreads to every "
                    "cell within a few steps, so the run stops here.";
        }
      }
    } catch (const std::exception &e) {
      failure = std::string("inference failed: ") + e.what();
    }
  }
  int length = static_cast<int>(failure.size());
  MPI_Bcast(&length, 1, MPI_INT, 0, m_comm);
  if (length > 0) {
    failure.resize(static_cast<std::size_t>(length));
    MPI_Bcast(failure.data(), length, MPI_CHAR, 0, m_comm);
    throw std::runtime_error("Network '" + m_layout.name + "' at step " +
                             std::to_string(step_index) + ": " + failure);
  }

  // 4. scatter
  for (std::size_t c = 0; c < m_layout.outputs.size(); ++c) {
    const auto src = root ? std::span<const double>(m_global_out)
                                .subspan(c * ncells, ncells)
                          : std::span<const double>();
    m_gather->scatter(src, m_prediction.get(m_layout.outputs[c]));
  }

  // 5. feed prognostic inputs from the raw prediction
  for (const auto &name :
       m_layout.inputs_from(fields::InputSource::Prognostic)) {
    const auto from = m_prediction.get(name);
    auto to = m_inputs.get(name);
    std::copy(from.begin(), from.end(), to.begin());
  }
}

} // namespace coupling
} // namespace emulator
