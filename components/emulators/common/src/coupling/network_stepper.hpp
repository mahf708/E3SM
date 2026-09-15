/**
 * @file network_stepper.hpp
 * @brief One step of a whole-grid network across a decomposed component.
 */

#ifndef E3SM_EMULATOR_COUPLING_NETWORK_STEPPER_HPP
#define E3SM_EMULATOR_COUPLING_NETWORK_STEPPER_HPP

#include <mpi.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "channel_layout.hpp"
#include "field_set.hpp"
#include "global_gather.hpp"
#include "inference_backend.hpp"

namespace emulator {
namespace coupling {

/**
 * @brief Runs a global network on the root rank for every rank.
 *
 * The component owns `inputs()` (one field per input channel, on its own
 * columns) and sets the coupled, boundary and forcing channels before each
 * step; prognostic channels are filled here, from the previous prediction.
 * step() then, on every rank, collectively:
 *
 *  1. gathers the inputs into a `[1, C_in, ny, nx]` tensor on the root, in
 *     layout channel order and the grid's own cell order;
 *  2. on the root, tells the backend the step (for a stochastic model's
 *     seed) and runs it into `[1, C_out, ny, nx]`;
 *  3. checks every output value is finite, and shares the verdict, so a bad
 *     step stops every rank instead of leaving the others waiting;
 *  4. scatters the outputs into `prediction()`;
 *  5. copies each prognostic input from the output of the same name.
 *
 * Step 5 uses the raw prediction, not the blended export shown to the
 * coupler, which lags the state by a fraction of a step.
 */
class NetworkStepper {
public:
  /**
   * @param backend used on the root only; may be null on other ranks, and
   *        must not be null on the root
   * @throws std::invalid_argument if the layout is invalid or nx*ny is not
   *         the grid size
   */
  NetworkStepper(fields::ChannelLayout layout, MPI_Comm comm,
                 const grid::GlobalGather &gather, int nx, int ny,
                 std::shared_ptr<inference::InferenceBackend> backend);

  /// One field per input channel, named as in the layout.
  fields::FieldSet &inputs() { return m_inputs; }
  const fields::FieldSet &inputs() const { return m_inputs; }
  /// One field per output channel, from the latest step.
  const fields::FieldSet &prediction() const { return m_prediction; }

  const fields::ChannelLayout &layout() const { return m_layout; }

  /**
   * @brief Advance the network one step.
   * @param step_index the component's counted step (LongStepClock), which a
   *        stochastic backend seeds from
   * @throws std::runtime_error on every rank if any output is not finite,
   *         naming the first bad channel
   */
  void step(std::int64_t step_index);

private:
  fields::ChannelLayout m_layout;
  MPI_Comm m_comm;
  const grid::GlobalGather *m_gather;
  int m_nx;
  int m_ny;
  std::shared_ptr<inference::InferenceBackend> m_backend;
  fields::FieldSet m_inputs;
  fields::FieldSet m_prediction;
  std::vector<double> m_global_in;  ///< root only
  std::vector<double> m_global_out; ///< root only
};

} // namespace coupling
} // namespace emulator

#endif // E3SM_EMULATOR_COUPLING_NETWORK_STEPPER_HPP
