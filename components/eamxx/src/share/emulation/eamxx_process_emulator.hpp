#ifndef EAMXX_PROCESS_EMULATOR_HPP
#define EAMXX_PROCESS_EMULATOR_HPP

#include "share/core/eamxx_types.hpp"

#include <ekat_pack.hpp>
#include <ekat_parameter_list.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace emulator {
namespace inference {
class InferenceBackend;
}
}

namespace scream
{

/*
 * Emulator of some of the process rates of a physics parameterization.
 *
 * A parameterization that can compute its process rates and apply them in two
 * steps (e.g., P3, see physics/p3/p3_process_rates.hpp) exposes, between the
 * two steps, its state and its rates, both by name. A ProcessEmulator runs a
 * model on some of these (its inputs), and overwrites some of the rates with
 * the model outputs. Which processes are emulated is only a matter of
 * configuration: any subset of the rates, from one to all of them.
 *
 * The model runs through one of the inference backends of components/emulators
 * (stub, python, libtorch). Each input and output is one tensor, named as the
 * quantity, with dims (ncol, nlev). For the python backend, they are the
 * entries of the inputs/outputs dicts passed to infer(); for the libtorch one,
 * they are the positional arguments of forward(), and the tuple it returns,
 * in the order of the configuration.
 *
 * Parameters (one sublist per emulator):
 *   backend:  stub | python | libtorch
 *   model_path: model file, passed to the backend
 *   inputs:   [names]  any state quantity, or any process rate (as computed
 *                      by the parameterization, before any emulator ran)
 *   outputs:  [names]  process rates to overwrite, and masks
 *   masks:              (optional) sublist mask_output: [rates it gates]
 *                       The gated rates are used only where the mask is > 0.5.
 *                       Rates not gated by any mask are always used.
 *   fallback_scale:     (optional) sublist rate: factor, applied to the rate
 *                       where its mask is <= 0.5 (default 1)
 *   mode:     replace | add   overwrite the rates, or add the outputs to them
 *   options:            (optional) sublist of backend options, see the
 *                       InferenceConfig of each backend (python_module, ...)
 *   verbose:  bool
 *
 * Data never leave the device except to go through the backend: inputs are
 * gathered from the (padded, packed) device views into contiguous staging
 * views on device, copied to host for the backend (a no-op on CPU), and the
 * outputs are merged into the packed rates on device.
 */
class ProcessEmulator
{
public:
  using Pack     = ekat::Pack<Real, SCREAM_PACK_SIZE>;
  using KT       = KokkosTypes<DefaultDevice>;
  using field_t  = KT::view_2d<const Pack>;  // (ncol, nlev_packs)
  using rates_t  = KT::view_3d<Pack>;        // (ncol, num_rates, nlev_packs)
  using backend_ptr = std::shared_ptr<emulator::inference::InferenceBackend>;

  // rate_names[i] is the name of rates(:,i,:)
  ProcessEmulator (const std::string& name, const ekat::ParameterList& params,
                   const std::vector<std::string>& rate_names,
                   const int ncol, const int nlev);

  // Same, with a given backend (the backend and model_path parameters are ignored)
  ProcessEmulator (const std::string& name, const ekat::ParameterList& params,
                   const std::vector<std::string>& rate_names,
                   const int ncol, const int nlev, const backend_ptr& backend);

  ~ProcessEmulator ();

  const std::string& name () const { return m_name; }
  const std::vector<std::string>& input_names  () const { return m_input_names; }
  const std::vector<std::string>& output_names () const { return m_output_names; }

  // Indices of the rates this emulator may change
  std::vector<int> emulated_rates () const;

  // Run the model and merge its outputs into rates.
  //  - state: the parameterization's state, by name
  //  - rates: the rates to update
  //  - original_rates: the rates as computed by the parameterization, for the
  //    inputs that are rates (may be the same as rates)
  void run (const std::map<std::string, field_t>& state,
            const rates_t& rates, const rates_t& original_rates);

private:
  void setup (const ekat::ParameterList& params, const std::vector<std::string>& rate_names);

  struct Output {
    int rate = -1;          // index of the rate it overwrites, or -1 for a mask
    int mask = -1;          // index (in outputs) of the mask gating it, or -1
    Real fallback_scale = 1;
  };

  using staging_t = KT::view_2d<double>;

  std::string               m_name;
  int                       m_ncol, m_nlev;
  bool                      m_add;
  std::vector<std::string>  m_input_names, m_output_names;
  std::vector<int>          m_input_rates;   // rate index of each input, or -1 for state
  std::vector<Output>       m_outputs;
  std::vector<staging_t>    m_in, m_out;     // (ncol, nlev), contiguous, on device
  std::vector<staging_t::host_mirror_type> m_in_h, m_out_h;
  backend_ptr               m_backend;
};

// Parse the backend name (stub | python | libtorch) of a ProcessEmulator sublist,
// and create the backend
ProcessEmulator::backend_ptr
create_process_emulator_backend (const std::string& name, const ekat::ParameterList& params);

} // namespace scream

#endif // EAMXX_PROCESS_EMULATOR_HPP
