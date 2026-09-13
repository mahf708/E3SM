/**
 * @file model_fields.hpp
 * @brief What an operator can see: the geometry, the step, and the named
 *        arrays of one rank, addressed by reference ("state.TS").
 */

#ifndef E3SM_EMULATOR_MODEL_MODEL_FIELDS_HPP
#define E3SM_EMULATOR_MODEL_MODEL_FIELDS_HPP

#include <mpi.h>

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "exchange.hpp"
#include "field_set.hpp"
#include "global_gather.hpp"
#include "horizontal_grid.hpp"
#include "interval_state.hpp"
#include "long_step_clock.hpp"

namespace emulator {
namespace model {

/// Where a model runs.
struct Geometry {
  MPI_Comm comm = MPI_COMM_NULL;
  /// The whole grid, identical on every rank; null for a component that has
  /// only its domain (the sea ice on the ocean's).  A network needs it.
  const grid::HorizontalGrid *grid = nullptr;
  grid::Decomposition decomp;
  std::shared_ptr<grid::GlobalGather> gather;
  std::vector<double> lat, lon, area; ///< this rank's cells
  std::vector<double> domain_mask;    ///< this rank's cells

  std::size_t num_local() const { return decomp.num_local(); }
  bool is_root() const { return gather->is_root(); }

  /// Collective.  An empty mask means every cell.
  static Geometry from_grid(MPI_Comm comm, const grid::HorizontalGrid &grid,
                            const grid::Decomposition &decomp,
                            std::vector<double> domain_mask = {});
  /// Collective.  For a component that takes another's domain.
  static Geometry from_domain(MPI_Comm comm, const grid::Domain &domain,
                              const grid::Decomposition &decomp);
};

/// The model time and where the clock is.
struct StepInfo {
  coupling::ModelTime now;
  coupling::LongStepClock::Step clock;
  int model_dt = 0; ///< seconds; 0 without a network
};

/**
 * @brief The named arrays of one rank at one step.
 *
 * - imports, exports: the coupler's fields the model reads and writes
 * - inputs: the network's input channels
 * - state: the outputs as the coupler sees them now (blended or held)
 * - prediction: the network's latest raw outputs
 * - upper: the upper bracket (the latest prediction, as carried)
 * - aux: arrays operators share with each other (insolation, say)
 * - statics: fields read once from the initial condition (masks)
 * - exchange: fields shared in-process with other emulated components
 */
struct Fields {
  const fields::FieldSet *imports = nullptr;
  fields::FieldSet *exports = nullptr;
  fields::FieldSet *inputs = nullptr;
  const fields::FieldSet *state = nullptr;
  const fields::FieldSet *prediction = nullptr;
  const coupling::BracketedState *upper = nullptr;
  fields::FieldSet *aux = nullptr;
  const fields::FieldSet *statics = nullptr;
  coupling::Exchange *exchange = nullptr;
};

/**
 * @brief A reference to one named array, written "<set>.<name>":
 *        imports, exports, inputs, state, prediction, upper, aux, statics,
 *        or exchange (whose names contain dots: "exchange.ocn.sst").
 */
class FieldRef {
public:
  enum class Set {
    Imports, Exports, Inputs, State, Prediction, Upper, Aux, Statics, Exchange
  };

  FieldRef() = default;
  /// @throws std::invalid_argument naming `where` on an unknown set
  static FieldRef parse(const std::string &text, const std::string &where);

  Set set() const { return m_set; }
  const std::string &name() const { return m_name; }
  std::string to_string() const { return m_text; }
  bool writable() const {
    return m_set == Set::Exports || m_set == Set::Inputs || m_set == Set::Aux;
  }

  /// @throws std::out_of_range naming the reference if it is not there
  std::span<const double> read(const Fields &f) const;
  /// @throws std::logic_error on a read-only set
  std::span<double> write(Fields &f) const;
  /// True if the array is there now.
  bool present(const Fields &f) const;

private:
  Set m_set = Set::Aux;
  std::string m_name;
  std::string m_text;
};

} // namespace model
} // namespace emulator

#endif // E3SM_EMULATOR_MODEL_MODEL_FIELDS_HPP
