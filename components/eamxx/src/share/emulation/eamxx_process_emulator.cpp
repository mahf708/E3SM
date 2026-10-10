#include "share/emulation/eamxx_process_emulator.hpp"

#include "create_inference_backend.hpp"
#include "inference_backend.hpp"
#include "inference_config.hpp"
#include "tensor.hpp"

#include <ekat_assert.hpp>
#include <ekat_std_utils.hpp>

#include <algorithm>
#include <sstream>

namespace scream
{

namespace {

std::string join (const std::vector<std::string>& v) {
  std::string s;
  for (const auto& e : v) {
    s += (s.empty() ? "" : ", ") + e;
  }
  return s;
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

emulator::inference::InferenceConfig
make_config (ekat::ParameterList params, const int n_in, const int n_out)
{
  emulator::inference::InferenceConfig config;
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

} // anonymous namespace

ProcessEmulator::backend_ptr
create_process_emulator_backend (const std::string& name, const ekat::ParameterList& params)
{
  using namespace emulator::inference;

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

ProcessEmulator::
ProcessEmulator (const std::string& name, const ekat::ParameterList& params,
                 const std::vector<std::string>& rate_names,
                 const int ncol, const int nlev)
 : ProcessEmulator(name, params, rate_names, ncol, nlev,
                   create_process_emulator_backend(name, params))
{
  // Nothing else to do
}

ProcessEmulator::
ProcessEmulator (const std::string& name, const ekat::ParameterList& params,
                 const std::vector<std::string>& rate_names,
                 const int ncol, const int nlev, const backend_ptr& backend)
 : m_name (name)
 , m_ncol (ncol)
 , m_nlev (nlev)
 , m_backend (backend)
{
  EKAT_REQUIRE_MSG (m_backend!=nullptr, "[ProcessEmulator] Error! Invalid backend for emulator '" + name + "'.\n");
  setup(params, rate_names);
}

ProcessEmulator::~ProcessEmulator ()
{
  if (m_backend) {
    m_backend->finalize();
  }
}

void ProcessEmulator::
setup (const ekat::ParameterList& params_in, const std::vector<std::string>& rate_names)
{
  auto params = params_in;
  const std::string prefix = "[ProcessEmulator] Error! In emulator '" + m_name + "', ";
  auto rate_index = [&](const std::string& n) {
    auto it = std::find(rate_names.begin(), rate_names.end(), n);
    return it==rate_names.end() ? -1 : static_cast<int>(it-rate_names.begin());
  };

  const auto mode = params.get<std::string>("mode", "replace");
  EKAT_REQUIRE_MSG (mode=="replace" or mode=="add", prefix + "mode must be 'replace' or 'add'.\n");
  m_add = mode=="add";

  m_input_names  = params.get<std::vector<std::string>>("inputs");
  m_output_names = params.get<std::vector<std::string>>("outputs");
  EKAT_REQUIRE_MSG (m_input_names.size()>0 and m_output_names.size()>0,
      prefix + "inputs and outputs cannot be empty.\n");

  for (const auto& n : m_input_names) {
    m_input_rates.push_back(rate_index(n));
  }

  // Outputs: rates, or masks
  const auto nout = m_output_names.size();
  m_outputs.resize(nout);
  std::vector<bool> is_mask(nout, false);
  auto output_index = [&](const std::string& n) {
    auto it = std::find(m_output_names.begin(), m_output_names.end(), n);
    return it==m_output_names.end() ? -1 : static_cast<int>(it-m_output_names.begin());
  };
  if (params.isSublist("masks")) {
    const auto& masks = params.sublist("masks");
    for (const auto& mname : masks.param_names()) {
      const int m = output_index(mname);
      EKAT_REQUIRE_MSG (m>=0, prefix + "mask '" + mname + "' is not an output.\n");
      EKAT_REQUIRE_MSG (rate_index(mname)<0, prefix + "mask '" + mname + "' is a process rate.\n");
      is_mask[m] = true;
      for (const auto& r : masks.get<std::vector<std::string>>(mname)) {
        const int o = output_index(r);
        EKAT_REQUIRE_MSG (o>=0, prefix + "rate '" + r + "' gated by mask '" + mname + "' is not an output.\n");
        EKAT_REQUIRE_MSG (m_outputs[o].mask<0, prefix + "rate '" + r + "' is gated by more than one mask.\n");
        m_outputs[o].mask = m;
      }
    }
  }
  for (size_t o=0; o<nout; ++o) {
    const auto& n = m_output_names[o];
    if (is_mask[o]) {
      continue;
    }
    m_outputs[o].rate = rate_index(n);
    EKAT_REQUIRE_MSG (m_outputs[o].rate>=0,
        prefix + "output '" + n + "' is neither a process rate nor a mask.\n"
        "  Process rates: " + join(rate_names) + "\n");
    for (size_t p=0; p<o; ++p) {
      EKAT_REQUIRE_MSG (m_outputs[p].rate!=m_outputs[o].rate, prefix + "output '" + n + "' is repeated.\n");
    }
  }
  if (params.isSublist("fallback_scale")) {
    const auto& fs = params.sublist("fallback_scale");
    for (const auto& rname : fs.param_names()) {
      const int o = output_index(rname);
      EKAT_REQUIRE_MSG (o>=0 and m_outputs[o].rate>=0 and m_outputs[o].mask>=0,
          prefix + "fallback_scale is set for '" + rname + "', which is not a rate gated by a mask.\n");
      m_outputs[o].fallback_scale = number(fs, rname);
    }
  }

  // Staging buffers, contiguous (ncol, nlev), on device and host
  for (const auto& n : m_input_names) {
    m_in.emplace_back("emu_in_" + n, m_ncol, m_nlev);
    m_in_h.push_back(Kokkos::create_mirror_view(m_in.back()));
  }
  for (const auto& n : m_output_names) {
    m_out.emplace_back("emu_out_" + n, m_ncol, m_nlev);
    m_out_h.push_back(Kokkos::create_mirror_view(m_out.back()));
  }
}

std::vector<int> ProcessEmulator::emulated_rates () const
{
  std::vector<int> r;
  for (const auto& o : m_outputs) {
    if (o.rate>=0) r.push_back(o.rate);
  }
  return r;
}

void ProcessEmulator::
run (const std::map<std::string, field_t>& state, const rates_t& rates, const rates_t& original_rates)
{
  using ExeSpace = KT::ExeSpace;
  using policy_t = Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<2>>;
  constexpr int N = Pack::n;

  const policy_t policy({0, 0}, {m_ncol, m_nlev});
  const int nrates = rates.extent_int(1);

  // 1. Gather the inputs into contiguous buffers
  for (size_t i=0; i<m_input_names.size(); ++i) {
    const auto in = m_in[i];
    const int r = m_input_rates[i];
    if (r>=0) {
      const auto src = original_rates;
      Kokkos::parallel_for("emu_gather_rate", policy, KOKKOS_LAMBDA (const int icol, const int ilev) {
        in(icol, ilev) = src(icol, r, ilev/N)[ilev%N];
      });
    } else {
      auto it = state.find(m_input_names[i]);
      if (it==state.end()) {
        std::vector<std::string> names;
        for (const auto& [n, v] : state) names.push_back(n);
        EKAT_ERROR_MSG ("[ProcessEmulator] Error! In emulator '" + m_name + "', input '" + m_input_names[i] +
                        "' is neither a state quantity nor a process rate.\n"
                        "  State quantities: " + join(names) + "\n");
      }
      const auto src = it->second;
      Kokkos::parallel_for("emu_gather_state", policy, KOKKOS_LAMBDA (const int icol, const int ilev) {
        in(icol, ilev) = src(icol, ilev/N)[ilev%N];
      });
    }
    Kokkos::deep_copy(m_in_h[i], in);
  }

  // 2. Run the model (on host memory)
  using namespace emulator::inference;
  TensorMap ins, outs;
  const std::vector<std::int64_t> dims = {m_ncol, m_nlev};
  for (size_t i=0; i<m_input_names.size(); ++i) {
    ins.wrap(m_input_names[i], static_cast<const double*>(m_in_h[i].data()), dims);
  }
  for (size_t o=0; o<m_output_names.size(); ++o) {
    Kokkos::deep_copy(m_out_h[o], 0);
    outs.wrap(m_output_names[o], m_out_h[o].data(), dims);
  }
  try {
    m_backend->infer(ins, outs);
  } catch (const std::exception& e) {
    EKAT_ERROR_MSG ("[ProcessEmulator] Error! Emulator '" + m_name + "' failed:\n" + e.what() + "\n");
  }
  for (size_t o=0; o<m_output_names.size(); ++o) {
    Kokkos::deep_copy(m_out[o], m_out_h[o]);
  }

  // 3. Merge the outputs into the rates
  const bool add = m_add;
  for (const auto& o : m_outputs) {
    if (o.rate<0) continue;
    EKAT_REQUIRE_MSG (o.rate<nrates, "[ProcessEmulator] Error! Rate index out of bounds.\n");
    const auto out  = m_out[&o-m_outputs.data()];
    const auto mask = o.mask>=0 ? m_out[o.mask] : staging_t();
    const bool has_mask = o.mask>=0;
    const Real fallback = o.fallback_scale;
    const int r = o.rate;
    Kokkos::parallel_for("emu_merge", policy, KOKKOS_LAMBDA (const int icol, const int ilev) {
      auto& v = rates(icol, r, ilev/N)[ilev%N];
      if (!has_mask || mask(icol, ilev) > 0.5) {
        v = add ? v + out(icol, ilev) : out(icol, ilev);
      } else {
        v *= fallback;
      }
    });
  }
  Kokkos::fence();
}

} // namespace scream
