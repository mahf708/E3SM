#include "share/emulation/eamxx_process_emulator.hpp"

#include "create_inference_backend.hpp"
#include "inference_backend.hpp"
#include "inference_config.hpp"
#include "tensor.hpp"

#include <ekat_assert.hpp>

#include <algorithm>
#include <sstream>
#include <type_traits>

namespace scream
{

namespace {

using namespace emulator::inference;

std::string join (const std::vector<std::string>& v) {
  std::string s;
  for (const auto& e : v) {
    s += (s.empty() ? "" : ", ") + e;
  }
  return s;
}

std::vector<std::string> names_of (const ProcessEmulator::arrays_t& m) {
  std::vector<std::string> n;
  for (const auto& [k, a] : m) n.push_back(k);
  return n;
}

// Backend options can be given in yaml as strings, numbers or bools
std::string option_as_string (const ekat::ParameterList& pl, const std::string& key) {
  if (pl.isType<std::string>(key)) return pl.get<std::string>(key);
  if (pl.isType<bool>(key))        return pl.get<bool>(key) ? "true" : "false";
  if (pl.isType<int>(key))         return std::to_string(pl.get<int>(key));
  if (pl.isType<double>(key)) {
    std::ostringstream ss;
    ss << pl.get<double>(key);
    return ss.str();
  }
  EKAT_ERROR_MSG ("[ProcessEmulator] Error! Option '" + key + "' must be a string, number or bool.\n");
}

double number (const ekat::ParameterList& pl, const std::string& key) {
  return pl.isType<int>(key) ? pl.get<int>(key) : pl.get<double>(key);
}

InferenceConfig make_config (ekat::ParameterList params, const int n_in, const int n_out)
{
  InferenceConfig config;
  config.model_path      = params.get<std::string>("model_path", "");
  config.input_channels  = n_in;
  config.output_channels = n_out;
  config.verbose         = params.get<bool>("verbose", false);
  if (params.isSublist("options")) {
    const auto& opts = params.sublist("options");
    for (const auto& key : opts.param_names()) {
      config.set(key, option_as_string(opts, key));
    }
  }
  return config;
}

// Where the arrays of the default device live
constexpr bool arrays_on_host =
  Kokkos::SpaceAccessibility<Kokkos::HostSpace, DefaultDevice::memory_space>::accessible;

TensorMemory array_memory () {
  TensorMemory m;
  if (not arrays_on_host) {
    m.space  = MemorySpace::DEVICE;
    m.device = Kokkos::device_id();
  }
  return m;
}

std::vector<std::int64_t> dims_of (const ProcessEmulator::Array& a) {
  if (a.per_column) return {static_cast<std::int64_t>(a.view.extent(0))};
  return {static_cast<std::int64_t>(a.view.extent(0)), static_cast<std::int64_t>(a.view.extent(1))};
}

std::vector<std::int64_t> strides_of (const ProcessEmulator::Array& a) {
  if (a.per_column) return {static_cast<std::int64_t>(a.view.stride(0))};
  return {static_cast<std::int64_t>(a.view.stride(0)), static_cast<std::int64_t>(a.view.stride(1))};
}

// Memory range [first, last) spanned by an array
std::pair<const Real*, const Real*> range_of (const ProcessEmulator::Array& a) {
  const auto n0 = a.view.extent(0), n1 = a.view.extent(1);
  if (n0==0 || n1==0) return {a.view.data(), a.view.data()};
  const auto last = (n0-1)*a.view.stride(0) + (n1-1)*a.view.stride(1);
  return {a.view.data(), a.view.data() + last + 1};
}

bool overlap (const std::pair<const Real*, const Real*>& a, const std::pair<const Real*, const Real*>& b) {
  return a.first < b.second && b.first < a.second;
}

// Whether two arrays share an element. Exact for arrays whose rows are
// contiguous and equally strided (slices of one packed view, fields of a
// group); otherwise, whether their memory ranges overlap.
bool overlap (const ProcessEmulator::Array& a, const ProcessEmulator::Array& b) {
  if (not overlap(range_of(a), range_of(b))) return false;
  const auto& va = a.view;
  const auto& vb = b.view;
  const auto s0 = static_cast<std::ptrdiff_t>(va.stride(0));
  const bool rows = va.extent(0)>1 and vb.extent(0)>1 and s0==static_cast<std::ptrdiff_t>(vb.stride(0)) and
                    (va.extent(1)<=1 or va.stride(1)==1) and (vb.extent(1)<=1 or vb.stride(1)==1) and
                    static_cast<std::ptrdiff_t>(va.extent(1))<=s0 and static_cast<std::ptrdiff_t>(vb.extent(1))<=s0;
  if (not rows) return true;
  // Rows of a: [i*s0, i*s0+na); rows of b: [d + j*s0, d + j*s0 + nb). With the
  // memory ranges overlapping, they share an element if b's rows start within
  // a row of a, or a's rows start within a row of b.
  const std::ptrdiff_t d  = vb.data() - va.data();
  const std::ptrdiff_t na = va.extent(1), nb = vb.extent(1);
  const std::ptrdiff_t r  = ((d % s0) + s0) % s0;
  return r < na or s0 - r < nb;
}

} // anonymous namespace

ProcessEmulator::backend_ptr
create_process_emulator_backend (const std::string& name, const ekat::ParameterList& params)
{
  const auto backend = params.get<std::string>("backend");
  BackendType type;
  if (backend=="stub") {
    type = BackendType::STUB;
  } else if (backend=="python") {
    type = BackendType::PYTHON;
  } else if (backend=="libtorch") {
    type = BackendType::LIBTORCH;
  } else {
    EKAT_ERROR_MSG ("[ProcessEmulator] Error! Unsupported backend '" + backend + "' for emulator '" +
                    name + "'.\n  Valid values: stub, python, libtorch.\n");
  }
  const int n_in  = params.get<std::vector<std::string>>("inputs").size();
  const int n_out = params.get<std::vector<std::string>>("outputs").size();
  try {
    return create_backend(type, make_config(params, n_in, n_out));
  } catch (const std::exception& e) {
    EKAT_ERROR_MSG ("[ProcessEmulator] Error! Could not create the " + backend + " backend of emulator '" +
                    name + "':\n" + e.what() + "\n");
  }
}

ProcessEmulator::Array ProcessEmulator::
strided (const Real* data, const int n0, const int n1, const int s0, const int s1, const bool per_column)
{
  Array a;
  a.view = view_t(const_cast<Real*>(data), Kokkos::LayoutStride(n0, s0, n1, s1));
  a.per_column = per_column;
  return a;
}

ProcessEmulator::
ProcessEmulator (const std::string& name, const ekat::ParameterList& params)
 : ProcessEmulator(name, params, create_process_emulator_backend(name, params))
{
  // Nothing else to do
}

ProcessEmulator::
ProcessEmulator (const std::string& name, const ekat::ParameterList& params, const backend_ptr& backend)
 : m_name (name)
 , m_backend (backend)
{
  EKAT_REQUIRE_MSG (m_backend!=nullptr, "[ProcessEmulator] Error! Invalid backend for emulator '" + name + "'.\n");
  setup(params);
}

ProcessEmulator::~ProcessEmulator ()
{
  if (m_backend) {
    m_backend->finalize();
  }
}

void ProcessEmulator::setup (const ekat::ParameterList& params_in)
{
  auto params = params_in;
  const std::string prefix = "[ProcessEmulator] Error! In emulator '" + m_name + "', ";

  const auto mode = params.get<std::string>("mode", "replace");
  EKAT_REQUIRE_MSG (mode=="replace" or mode=="add", prefix + "mode must be 'replace' or 'add'.\n");
  m_add = mode=="add";

  m_input_names  = params.get<std::vector<std::string>>("inputs");
  m_output_names = params.get<std::vector<std::string>>("outputs");
  EKAT_REQUIRE_MSG (m_input_names.size()>0 and m_output_names.size()>0,
      prefix + "inputs and outputs cannot be empty.\n");
  for (size_t o=0; o<m_output_names.size(); ++o) {
    for (size_t p=0; p<o; ++p) {
      EKAT_REQUIRE_MSG (m_output_names[p]!=m_output_names[o],
          prefix + "output '" + m_output_names[o] + "' is repeated.\n");
    }
  }

  auto output_index = [&](const std::string& n) {
    auto it = std::find(m_output_names.begin(), m_output_names.end(), n);
    return it==m_output_names.end() ? -1 : static_cast<int>(it-m_output_names.begin());
  };
  m_outputs.resize(m_output_names.size());
  if (params.isSublist("masks")) {
    const auto& masks = params.sublist("masks");
    for (const auto& mname : masks.param_names()) {
      const int m = output_index(mname);
      EKAT_REQUIRE_MSG (m>=0, prefix + "mask '" + mname + "' is not an output.\n");
      m_outputs[m].is_mask = true;
    }
    for (const auto& mname : masks.param_names()) {
      const int m = output_index(mname);
      for (const auto& t : masks.get<std::vector<std::string>>(mname)) {
        const int o = output_index(t);
        EKAT_REQUIRE_MSG (o>=0, prefix + "'" + t + "', gated by mask '" + mname + "', is not an output.\n");
        EKAT_REQUIRE_MSG (not m_outputs[o].is_mask, prefix + "mask '" + t + "' is gated by a mask.\n");
        EKAT_REQUIRE_MSG (m_outputs[o].mask<0, prefix + "'" + t + "' is gated by more than one mask.\n");
        m_outputs[o].mask = m;
      }
    }
  }
  if (params.isSublist("fallback_scale")) {
    const auto& fs = params.sublist("fallback_scale");
    for (const auto& t : fs.param_names()) {
      const int o = output_index(t);
      EKAT_REQUIRE_MSG (o>=0 and m_outputs[o].mask>=0,
          prefix + "fallback_scale is set for '" + t + "', which is not an output gated by a mask.\n");
      m_outputs[o].fallback_scale = number(fs, t);
    }
  }
}

std::vector<std::string> ProcessEmulator::target_names () const
{
  std::vector<std::string> t;
  for (size_t o=0; o<m_outputs.size(); ++o) {
    if (not m_outputs[o].is_mask) t.push_back(m_output_names[o]);
  }
  return t;
}

ProcessEmulator::Buffer& ProcessEmulator::
buffer (std::map<std::string, Buffer>& buffers, const std::string& name, const int n0, const int n1)
{
  auto& b = buffers[name];
  if (b.dev.extent_int(0)!=n0 or b.dev.extent_int(1)!=n1) {
    b.dev  = KT::view_2d<double>("emu_" + m_name + "_" + name, n0, n1);
    b.host = Kokkos::create_mirror_view(b.dev);
  }
  return b;
}

void ProcessEmulator::run (const arrays_t& inputs, const arrays_t& targets)
{
  using policy_t = Kokkos::MDRangePolicy<KT::ExeSpace, Kokkos::Rank<2>>;
  const std::string prefix = "[ProcessEmulator] Error! In emulator '" + m_name + "', ";

  // Arrays go to the backend in place if it accepts their memory, and they are doubles
  const auto memory = array_memory();
  const bool in_place = std::is_same<Real, double>::value and m_backend->accepts(memory.space);
  // Otherwise, through buffers: on device if the backend accepts device memory
  TensorMemory buf_memory;
  const bool buffers_on_device = not arrays_on_host and m_backend->accepts(MemorySpace::DEVICE);
  if (buffers_on_device) {
    buf_memory = memory;
  }
  m_num_in_place = m_num_buffered = 0;

  // The (device) data the backend reads must be ready
  Kokkos::fence();

  // 1. Inputs
  TensorMap ins;
  std::vector<const Array*> in_place_arrays;
  for (const auto& n : m_input_names) {
    auto it = inputs.find(n);
    EKAT_REQUIRE_MSG (it!=inputs.end(), prefix + "input '" + n + "' is not available.\n"
                      "  Available inputs: " + join(names_of(inputs)) + "\n");
    const auto& a = it->second;
    if (in_place) {
      ins.wrap(n, static_cast<const double*>(static_cast<const void*>(a.view.data())),
               dims_of(a), strides_of(a), memory);
      in_place_arrays.push_back(&a);
      ++m_num_in_place;
    } else {
      const int n0 = a.view.extent(0), n1 = a.view.extent(1);
      auto& b = buffer(m_in_buffers, n, n0, n1);
      const auto src = a.view;
      const auto dst = b.dev;
      Kokkos::parallel_for("emu_gather", policy_t({0,0},{n0,n1}), KOKKOS_LAMBDA (const int i, const int k) {
        dst(i,k) = src(i,k);
      });
      const double* data = b.dev.data();
      if (not buffers_on_device) {
        Kokkos::deep_copy(b.host, b.dev);
        data = b.host.data();
      }
      ins.wrap(n, data, dims_of(a), {}, buf_memory);
      ++m_num_buffered;
    }
  }

  // 2. Outputs: targets written in place by the backend, or buffers merged afterwards
  TensorMap outs;
  std::vector<bool> direct(m_output_names.size(), false);
  std::vector<const Array*> target_of(m_output_names.size(), nullptr);
  std::vector<std::pair<int,int>> out_dims(m_output_names.size());
  for (size_t o=0; o<m_output_names.size(); ++o) {
    const auto& n = m_output_names[o];
    const Array* t = nullptr;
    if (not m_outputs[o].is_mask) {
      auto it = targets.find(n);
      EKAT_REQUIRE_MSG (it!=targets.end(), prefix + "output '" + n + "' is not a mask, nor an available target.\n"
                        "  Available targets: " + join(names_of(targets)) + "\n");
      t = &it->second;
      target_of[o] = t;
    }
    // Masks have the shape of the first target they gate
    const Array* shape = t;
    if (shape==nullptr) {
      for (size_t p=0; p<m_outputs.size() and shape==nullptr; ++p) {
        if (m_outputs[p].mask==static_cast<int>(o)) {
          auto it = targets.find(m_output_names[p]);
          if (it!=targets.end()) shape = &it->second;
        }
      }
      EKAT_REQUIRE_MSG (shape!=nullptr, prefix + "mask '" + n + "' gates no available target.\n");
    }
    out_dims[o] = {static_cast<int>(shape->view.extent(0)), static_cast<int>(shape->view.extent(1))};

    bool aliased = false;
    if (t!=nullptr) {
      for (const auto* a : in_place_arrays) aliased |= overlap(*a, *t);
    }
    direct[o] = in_place and t!=nullptr and m_outputs[o].mask<0 and not m_add and not aliased;
    if (direct[o]) {
      outs.wrap(n, static_cast<double*>(static_cast<void*>(t->view.data())),
                dims_of(*t), strides_of(*t), memory);
      ++m_num_in_place;
    } else {
      auto& b = buffer(m_out_buffers, n, out_dims[o].first, out_dims[o].second);
      if (t!=nullptr and not m_add) {
        // Replace mode: start from the target, so that what the model does not
        // write keeps its value, as it does when the target is passed in place
        const auto src = t->view;
        const auto dst = b.dev;
        Kokkos::parallel_for("emu_seed", policy_t({0,0},{out_dims[o].first, out_dims[o].second}),
                             KOKKOS_LAMBDA (const int i, const int k) {
          dst(i,k) = src(i,k);
        });
      } else {
        // Masks start off, and add mode adds what the model writes
        Kokkos::deep_copy(b.dev, 0);
      }
      double* data = b.dev.data();
      if (not buffers_on_device) {
        Kokkos::deep_copy(b.host, b.dev);
        data = b.host.data();
      }
      outs.wrap(n, data, dims_of(*shape), {}, buf_memory);
      ++m_num_buffered;
    }
  }
  Kokkos::fence();

  // 3. Run the model
  try {
    m_backend->infer(ins, outs);
  } catch (const std::exception& e) {
    EKAT_ERROR_MSG ("[ProcessEmulator] Error! Emulator '" + m_name + "' failed:\n" + e.what() + "\n");
  }

  // 4. Merge the buffered outputs into their targets
  if (not buffers_on_device) {
    for (size_t o=0; o<m_output_names.size(); ++o) {
      if (not direct[o]) {
        auto& b = m_out_buffers.at(m_output_names[o]);
        Kokkos::deep_copy(b.dev, b.host);
      }
    }
  }
  const bool add = m_add;
  for (size_t o=0; o<m_output_names.size(); ++o) {
    if (direct[o] or target_of[o]==nullptr) continue;
    const auto& oo = m_outputs[o];
    const auto tgt  = target_of[o]->view;
    const auto out  = m_out_buffers.at(m_output_names[o]).dev;
    const auto mask = oo.mask>=0 ? m_out_buffers.at(m_output_names[oo.mask]).dev : KT::view_2d<double>();
    EKAT_REQUIRE_MSG (oo.mask<0 or (mask.extent(0)==tgt.extent(0) and mask.extent(1)==tgt.extent(1)),
        prefix + "mask '" + m_output_names[oo.mask] + "' and '" + m_output_names[o] + "' differ in shape.\n");
    const bool has_mask = oo.mask>=0;
    const Real fallback = oo.fallback_scale;
    Kokkos::parallel_for("emu_merge", policy_t({0,0},{out_dims[o].first, out_dims[o].second}),
                         KOKKOS_LAMBDA (const int i, const int k) {
      auto& v = tgt(i,k);
      if (!has_mask || mask(i,k) > 0.5) {
        v = add ? v + out(i,k) : static_cast<Real>(out(i,k));
      } else {
        v *= fallback;
      }
    });
  }
  Kokkos::fence();
}

} // namespace scream
