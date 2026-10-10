#include <catch2/catch.hpp>

#include "share/emulation/eamxx_process_emulator.hpp"

#include "inference_backend.hpp"
#include "tensor.hpp"

#include <functional>
#include <map>

namespace scream {

namespace {

using namespace emulator::inference;

// A backend running a C++ function on host memory, to test the plumbing
// without ML libraries. Tensors may be strided: elements are read and written
// through their strides.
class FunctionBackend : public InferenceBackend {
public:
  using fn_t = std::function<void(const TensorMap&, TensorMap&)>;
  explicit FunctionBackend (const fn_t& f) : InferenceBackend(InferenceConfig()), m_f(f) {}
  using InferenceBackend::infer;
  bool infer (const TensorMap& inputs, TensorMap& outputs) override { m_f(inputs, outputs); return true; }
  void finalize () override {}
  std::string name () const override { return "Function"; }
private:
  fn_t m_f;
};

const Tensor& find (const TensorMap& m, const std::string& n) {
  for (const auto& t : m) if (t.name()==n) return t;
  throw std::runtime_error("no tensor " + n);
}
Tensor& find (TensorMap& m, const std::string& n) {
  for (auto& t : m) if (t.name()==n) return t;
  throw std::runtime_error("no tensor " + n);
}

// Element (i,k) of a rank-2 tensor, or (i) of a rank-1 one
double get (const Tensor& t, const int i, const int k = 0) {
  const auto& s = t.strides();
  return t.cdata()[i*s[0] + (s.size()>1 ? k*s[1] : 0)];
}
void set (Tensor& t, const int i, const int k, const double v) {
  const auto& s = t.strides();
  t.data()[i*s[0] + (s.size()>1 ? k*s[1] : 0)] = v;
}

} // anonymous namespace

TEST_CASE ("process_emulator") {
  using PE   = ProcessEmulator;
  using Pack = ekat::Pack<Real, SCREAM_PACK_SIZE>;
  using KT   = KokkosTypes<DefaultDevice>;
  constexpr int N = Pack::n;

  // nlev not a multiple of the pack size, to exercise the padding
  const int ncol = 3, nlev = 2*N + 1, npack = ekat::npack<Pack>(nlev);

  // State x(i,k) = i + k/10 (packed); rates a = 100, b = 200, c = 300 (packed, in one 3d view,
  // also in the padding); a per-column quantity p(i) = i
  KT::view_2d<Pack> x("x", ncol, npack);
  KT::view_3d<Pack> rates("rates", ncol, 3, npack);
  KT::view_1d<Real> p("p", ncol);
  auto reset = [&]() {
    auto x_h = Kokkos::create_mirror_view(x);
    auto r_h = Kokkos::create_mirror_view(rates);
    auto p_h = Kokkos::create_mirror_view(p);
    for (int i=0; i<ncol; ++i) {
      p_h(i) = i;
      for (int k=0; k<npack*N; ++k) {
        x_h(i, k/N)[k%N] = i + 0.1*k;
        for (int r=0; r<3; ++r) {
          r_h(i, r, k/N)[k%N] = 100*(r+1);
        }
      }
    }
    Kokkos::deep_copy(x, x_h);
    Kokkos::deep_copy(rates, r_h);
    Kokkos::deep_copy(p, p_h);
  };
  auto rate = [&](const int i, const int r, const int k) {
    auto r_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rates);
    return r_h(i, r, k/N)[k%N];
  };
  const PE::arrays_t inputs = {{"x", PE::array(x, nlev)}, {"p", PE::array(p)},
                               {"c", PE::array(rates, 2, nlev)}};
  const PE::arrays_t targets = {{"a", PE::array(rates, 0, nlev)}, {"b", PE::array(rates, 1, nlev)},
                                {"c", PE::array(rates, 2, nlev)}};
  constexpr bool on_host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, KT::MemSpace>::accessible;
  constexpr bool is_double = std::is_same<Real, double>::value;

  SECTION ("replace, with a mask and a fallback") {
    // a = 2x everywhere; b = x+1 where x > 1, else b *= 0.5; c is not emulated
    ekat::ParameterList pl("emu");
    pl.set<std::vector<std::string>>("inputs", {"x"});
    pl.set<std::vector<std::string>>("outputs", {"a", "b", "m"});
    pl.sublist("masks").set<std::vector<std::string>>("m", {"b"});
    pl.sublist("fallback_scale").set<double>("b", 0.5);
    auto backend = std::make_shared<FunctionBackend>([&](const TensorMap& in, TensorMap& out) {
      const auto& xi = find(in, "x");
      auto &a = find(out, "a"), &b = find(out, "b"), &m = find(out, "m");
      REQUIRE (xi.dims()==std::vector<std::int64_t>{ncol, nlev});
      for (int i=0; i<ncol; ++i) {
        for (int k=0; k<nlev; ++k) {
          set(a, i, k, 2*get(xi, i, k));
          set(b, i, k, get(xi, i, k) + 1);
          set(m, i, k, get(xi, i, k) > 1 ? 1 : 0);
        }
      }
    });
    PE emu("emu", pl, backend);
    REQUIRE (emu.target_names()==std::vector<std::string>{"a", "b"});

    reset();
    emu.run(inputs, targets);
    if (on_host and is_double) {
      // x and a in place, b and m through buffers (b is gated by m)
      REQUIRE (emu.num_in_place()==2);
      REQUIRE (emu.num_buffered()==2);
    }
    for (int i=0; i<ncol; ++i) {
      for (int k=0; k<nlev; ++k) {
        const double xv = i + 0.1*k;
        REQUIRE (rate(i,0,k) == Approx(2*xv));
        REQUIRE (rate(i,1,k) == Approx(xv > 1 ? xv + 1 : 100.0));
        REQUIRE (rate(i,2,k) == 300);
      }
      // The padding is untouched
      for (int k=nlev; k<npack*N; ++k) {
        REQUIRE (rate(i,0,k) == 100);
        REQUIRE (rate(i,1,k) == 200);
      }
    }
  }

  SECTION ("add, with a rate as input, and a per-column input") {
    // a += 3c + p
    ekat::ParameterList pl("emu");
    pl.set<std::vector<std::string>>("inputs", {"c", "p"});
    pl.set<std::vector<std::string>>("outputs", {"a"});
    pl.set<std::string>("mode", "add");
    auto backend = std::make_shared<FunctionBackend>([&](const TensorMap& in, TensorMap& out) {
      const auto& c = find(in, "c");
      const auto& pc = find(in, "p");
      REQUIRE (pc.dims()==std::vector<std::int64_t>{ncol});
      auto& a = find(out, "a");
      for (int i=0; i<ncol; ++i)
        for (int k=0; k<nlev; ++k)
          set(a, i, k, 3*get(c, i, k) + get(pc, i));
    });
    PE emu("emu", pl, backend);

    reset();
    emu.run(inputs, targets);
    for (int i=0; i<ncol; ++i) {
      for (int k=0; k<nlev; ++k) {
        REQUIRE (rate(i,0,k) == 100 + 3*300 + i);
      }
    }
  }

  SECTION ("an output that is also an input") {
    // c = 2c: c cannot be written in place while the backend reads it
    ekat::ParameterList pl("emu");
    pl.set<std::vector<std::string>>("inputs", {"c"});
    pl.set<std::vector<std::string>>("outputs", {"c"});
    auto backend = std::make_shared<FunctionBackend>([&](const TensorMap& in, TensorMap& out) {
      const auto& c_in = find(in, "c");
      auto& c_out = find(out, "c");
      REQUIRE (c_in.cdata()!=c_out.cdata());
      for (int i=0; i<ncol; ++i)
        for (int k=0; k<nlev; ++k)
          set(c_out, i, k, 2*get(c_in, i, k));
    });
    PE emu("emu", pl, backend);

    reset();
    emu.run(inputs, targets);
    for (int i=0; i<ncol; ++i) {
      for (int k=0; k<nlev; ++k) {
        REQUIRE (rate(i,2,k) == 600);
      }
    }
  }

  SECTION ("an output the model does not write") {
    // The model writes a = 1 and skips b. In replace mode, b keeps its value,
    // whether it is passed in place (no alias) or through a buffer (b is also an input)
    for (const bool b_is_input : {false, true}) {
      ekat::ParameterList pl("emu");
      pl.set<std::vector<std::string>>("inputs", b_is_input ? std::vector<std::string>{"x", "b"}
                                                            : std::vector<std::string>{"x"});
      pl.set<std::vector<std::string>>("outputs", {"a", "b"});
      auto backend = std::make_shared<FunctionBackend>([&](const TensorMap&, TensorMap& out) {
        auto& a = find(out, "a");
        for (int i=0; i<ncol; ++i)
          for (int k=0; k<nlev; ++k)
            set(a, i, k, 1);
      });
      PE emu("emu", pl, backend);
      PE::arrays_t ins = inputs;
      ins["b"] = PE::array(rates, 1, nlev);
      reset();
      emu.run(ins, targets);
      if (on_host and is_double) {
        REQUIRE (emu.num_buffered() == (b_is_input ? 1 : 0));
      }
      for (int i=0; i<ncol; ++i) {
        for (int k=0; k<nlev; ++k) {
          REQUIRE (rate(i,0,k) == 1);
          REQUIRE (rate(i,1,k) == 200);
        }
      }
    }
  }

  SECTION ("tensors") {
    // Inputs are named, in the configured order, read-only, (ncol, nlev)
    ekat::ParameterList pl("emu");
    pl.set<std::vector<std::string>>("inputs", {"x", "c"});
    pl.set<std::vector<std::string>>("outputs", {"a"});
    auto backend = std::make_shared<FunctionBackend>([&](const TensorMap& in, TensorMap& out) {
      std::vector<std::string> names;
      for (const auto& t : in) {
        names.push_back(t.name());
        REQUIRE (t.dims()==std::vector<std::int64_t>{ncol, nlev});
        REQUIRE (not t.writable());
        if (on_host and is_double) {
          // Packed views, passed in place, with their padding as row stride
          REQUIRE (not t.contiguous());
          REQUIRE (t.strides()[1]==1);
        }
      }
      REQUIRE (names==std::vector<std::string>{"x", "c"});
      REQUIRE (get(find(in, "x"), 1, 2) == Approx(1 + 0.2));
      auto& a = find(out, "a");
      REQUIRE (a.writable());
      for (int i=0; i<ncol; ++i)
        for (int k=0; k<nlev; ++k)
          set(a, i, k, 0);
    });
    PE emu("emu", pl, backend);
    reset();
    emu.run(inputs, targets);
    REQUIRE (rate(0,0,0) == 0);
  }

  SECTION ("errors") {
    auto backend = std::make_shared<FunctionBackend>([](const TensorMap&, TensorMap&) {});
    ekat::ParameterList pl("emu");
    pl.set<std::vector<std::string>>("inputs", {"x"});

    // A mask gating something that is not an output
    pl.set<std::vector<std::string>>("outputs", {"a", "m"});
    pl.sublist("masks").set<std::vector<std::string>>("m", {"b"});
    REQUIRE_THROWS (PE("emu", pl, backend));

    reset();
    // An output that is neither a mask nor a target
    ekat::ParameterList q("emu");
    q.set<std::vector<std::string>>("inputs", {"x"});
    q.set<std::vector<std::string>>("outputs", {"z"});
    PE emu_q("emu", q, backend);
    REQUIRE_THROWS (emu_q.run(inputs, targets));

    // An input that does not exist
    ekat::ParameterList r("emu");
    r.set<std::vector<std::string>>("inputs", {"y"});
    r.set<std::vector<std::string>>("outputs", {"a"});
    PE emu_r("emu", r, backend);
    REQUIRE_THROWS (emu_r.run(inputs, targets));
  }
}

} // namespace scream
