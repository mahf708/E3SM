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
 * An emulator: a model that computes some named quantities from others.
 *
 * It knows nothing about the physics it emulates. The caller gives it named
 * input arrays and named target arrays, and the emulator overwrites (or adds
 * to) some of the targets with the outputs of its model. The arrays can be the
 * process rates of a parameterization (see P3), the intermediate quantities of
 * one (see SHOC), or the fields of a whole atmosphere process.
 *
 * The model runs through one of the inference backends of components/emulators
 * (stub, python, libtorch). Each input and output is one tensor, named as the
 * quantity, with dims (ncol, nlev), or (ncol) for per-column quantities. For
 * the python backend they are the entries of the inputs/outputs dicts of
 * infer(); for libtorch, the positional arguments of forward() and the tuple
 * it returns, in the order of the configuration.
 *
 * No copies: arrays are passed to the backend in place, with their strides
 * (e.g., padded or packed views), when the backend accepts their memory
 * space (host, or device for a libtorch module on the GPU or a python model
 * with device_arrays), and Real is double. Otherwise they go through
 * contiguous double buffers, on device first, and on host only if the
 * backend needs host memory. A target is written in place by the backend
 * when nothing else needs to happen to it (no mask, replace mode, no
 * overlap with an input); otherwise the merge runs on device.
 *
 * Parameters (one sublist per emulator):
 *   backend:  stub | python | libtorch
 *   model_path: model file, passed to the backend
 *   inputs:   [names]  input arrays
 *   outputs:  [names]  target arrays to overwrite, and masks
 *   masks:              (optional) sublist mask_output: [targets it gates]
 *                       Gated targets change only where the mask is > 0.5.
 *   fallback_scale:     (optional) sublist target: factor, applied to the
 *                       target where its mask is <= 0.5 (default 1)
 *   mode:     replace | add   overwrite the targets, or add the outputs to them
 *   physics:  run | skip      (default run) with skip, the physics that computes
 *                       the targets is not run where the cut point allows it
 *                       (it is then an error to read the targets as inputs, or
 *                       to use masks or mode add)
 *   options:            (optional) sublist of backend options, see the
 *                       InferenceConfig of each backend
 *   verbose:  bool
 */
class ProcessEmulator
{
public:
  using KT     = KokkosTypes<DefaultDevice>;
  using view_t = Kokkos::View<Real**, Kokkos::LayoutStride, DefaultDevice::memory_space,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using backend_ptr = std::shared_ptr<emulator::inference::InferenceBackend>;

  // A named quantity: (ncol, nlev), or (ncol) viewed as (ncol, 1), with any strides
  struct Array {
    view_t view;
    bool per_column = false;
  };
  using arrays_t = std::map<std::string, Array>;

  ProcessEmulator (const std::string& name, const ekat::ParameterList& params);

  // Same, with a given backend (the backend and model_path parameters are ignored)
  ProcessEmulator (const std::string& name, const ekat::ParameterList& params,
                   const backend_ptr& backend);

  ~ProcessEmulator ();

  const std::string& name () const { return m_name; }
  const std::vector<std::string>& input_names  () const { return m_input_names; }
  const std::vector<std::string>& output_names () const { return m_output_names; }

  // The outputs that are targets (i.e., not masks)
  std::vector<std::string> target_names () const;

  // Whether the physics that computes the targets may be skipped (physics: skip):
  // the emulator replaces it, instead of overwriting what it computed
  bool skips_physics () const { return m_skip_physics; }

  // Run the model on the inputs, and update the targets with its outputs.
  // Inputs and targets may be the same arrays.
  void run (const arrays_t& inputs, const arrays_t& targets);

  // How many arrays the last run passed to the backend in place, and through buffers
  int num_in_place () const { return m_num_in_place; }
  int num_buffered () const { return m_num_buffered; }

  // ---- Arrays from common views ---- //

  // A packed (ncol, nlev_packs) view, of which the first nlev levels are used
  template<typename PackT, typename... Props>
  static Array array (const Kokkos::View<PackT**, Props...>& v, const int nlev) {
    using value_t = typename std::remove_const<PackT>::type;
    static_assert(sizeof(value_t)==value_t::n*sizeof(Real), "Unexpected pack type.");
    return strided(reinterpret_cast<const Real*>(v.data()), v.extent(0), nlev,
                   v.stride(0)*value_t::n, 1, false);
  }

  // Slice i of a packed (ncol, n, nlev_packs) view
  template<typename PackT, typename... Props>
  static Array array (const Kokkos::View<PackT***, Props...>& v, const int i, const int nlev) {
    using value_t = typename std::remove_const<PackT>::type;
    static_assert(sizeof(value_t)==value_t::n*sizeof(Real), "Unexpected pack type.");
    return strided(reinterpret_cast<const Real*>(v.data() + i*v.stride(1)), v.extent(0), nlev,
                   v.stride(0)*value_t::n, 1, false);
  }

  // A (ncol, nlev) view of reals, possibly padded
  template<typename... Props>
  static Array array (const Kokkos::View<Real**, Props...>& v) {
    return strided(v.data(), v.extent(0), v.extent(1), v.stride(0), v.stride(1), false);
  }
  template<typename... Props>
  static Array array (const Kokkos::View<const Real**, Props...>& v) {
    return strided(v.data(), v.extent(0), v.extent(1), v.stride(0), v.stride(1), false);
  }

  // A (ncol) view of reals
  template<typename... Props>
  static Array array (const Kokkos::View<Real*, Props...>& v) {
    return strided(v.data(), v.extent(0), 1, v.stride(0), 1, true);
  }
  template<typename... Props>
  static Array array (const Kokkos::View<const Real*, Props...>& v) {
    return strided(v.data(), v.extent(0), 1, v.stride(0), 1, true);
  }

  static Array strided (const Real* data, const int n0, const int n1,
                        const int s0, const int s1, const bool per_column);

private:
  void setup (const ekat::ParameterList& params);

  struct Output {
    bool is_mask = false;
    int mask = -1;          // index (in outputs) of the mask gating it, or -1
    Real fallback_scale = 1;
  };

  // Contiguous double buffer, on device, with a host mirror
  struct Buffer {
    KT::view_2d<double> dev;
    KT::view_2d<double>::host_mirror_type host;
  };
  Buffer& buffer (std::map<std::string, Buffer>& buffers, const std::string& name,
                  const int n0, const int n1);

  std::string               m_name;
  bool                      m_add;
  bool                      m_skip_physics = false;
  std::vector<std::string>  m_input_names, m_output_names;
  std::vector<Output>       m_outputs;
  std::map<std::string, Buffer> m_in_buffers, m_out_buffers;
  backend_ptr               m_backend;
  int m_num_in_place = 0, m_num_buffered = 0;
};

// Create the backend (stub | python | libtorch) of a ProcessEmulator sublist
ProcessEmulator::backend_ptr
create_process_emulator_backend (const std::string& name, const ekat::ParameterList& params);

} // namespace scream

#endif // EAMXX_PROCESS_EMULATOR_HPP
