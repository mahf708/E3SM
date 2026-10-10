#include <catch2/catch.hpp>

#include "share/emulation/eamxx_process_emulator.hpp"

#include "inference_backend.hpp"
#include "tensor.hpp"

#include <functional>
#include <map>

namespace scream {

namespace {

using namespace emulator::inference;

// A backend running a C++ function, to test the plumbing without ML libraries
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

const double* find (const TensorMap& m, const std::string& n) {
  for (const auto& t : m) if (t.name()==n) return t.cdata();
  return nullptr;
}
double* find (TensorMap& m, const std::string& n) {
  for (auto& t : m) if (t.name()==n) return t.data();
  return nullptr;
}

} // anonymous namespace

TEST_CASE ("process_emulator") {
  using PE   = ProcessEmulator;
  using Pack = PE::Pack;
  constexpr int N = Pack::n;

  // nlev not a multiple of the pack size, to exercise the padding
  const int ncol = 3, nlev = 2*N + 1, npack = ekat::npack<Pack>(nlev);
  const std::vector<std::string> rate_names = {"a", "b", "c"};

  // State x(i,k) = i + k/10; rates a = 100, b = 200, c = 300 (also in the padding)
  KokkosTypes<DefaultDevice>::view_2d<Pack> x("x", ncol, npack);
  PE::rates_t rates("rates", ncol, 3, npack);
  auto reset = [&]() {
    auto x_h = Kokkos::create_mirror_view(x);
    auto r_h = Kokkos::create_mirror_view(rates);
    for (int i=0; i<ncol; ++i) {
      for (int k=0; k<npack*N; ++k) {
        x_h(i, k/N)[k%N] = i + 0.1*k;
        for (int r=0; r<3; ++r) {
          r_h(i, r, k/N)[k%N] = 100*(r+1);
        }
      }
    }
    Kokkos::deep_copy(x, x_h);
    Kokkos::deep_copy(rates, r_h);
  };
  const std::map<std::string, PE::field_t> state = {{"x", x}};
  auto rate = [&](const int i, const int r, const int k) {
    auto r_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rates);
    return r_h(i, r, k/N)[k%N];
  };

  SECTION ("replace, with a mask and a fallback") {
    // a = 2x everywhere; b = x+1 where x > 1, else b *= 0.5; c is not emulated
    ekat::ParameterList p("emu");
    p.set<std::vector<std::string>>("inputs", {"x"});
    p.set<std::vector<std::string>>("outputs", {"a", "b", "m"});
    p.sublist("masks").set<std::vector<std::string>>("m", {"b"});
    p.sublist("fallback_scale").set<double>("b", 0.5);
    auto backend = std::make_shared<FunctionBackend>([&](const TensorMap& in, TensorMap& out) {
      const double* xi = find(in, "x");
      double *a = find(out, "a"), *b = find(out, "b"), *m = find(out, "m");
      for (int n=0; n<ncol*nlev; ++n) {
        a[n] = 2*xi[n];
        b[n] = xi[n] + 1;
        m[n] = xi[n] > 1 ? 1 : 0;
      }
    });
    PE emu("emu", p, rate_names, ncol, nlev, backend);
    REQUIRE (emu.emulated_rates()==std::vector<int>{0, 1});

    reset();
    emu.run(state, rates, rates);
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

  SECTION ("add, with a rate as input") {
    // a += 3c, using the rates as computed by the parameterization
    ekat::ParameterList p("emu");
    p.set<std::vector<std::string>>("inputs", {"c"});
    p.set<std::vector<std::string>>("outputs", {"a"});
    p.set<std::string>("mode", "add");
    auto backend = std::make_shared<FunctionBackend>([&](const TensorMap& in, TensorMap& out) {
      const double* c = find(in, "c");
      double* a = find(out, "a");
      for (int n=0; n<ncol*nlev; ++n) a[n] = 3*c[n];
    });
    PE emu("emu", p, rate_names, ncol, nlev, backend);

    reset();
    emu.run(state, rates, rates);
    for (int i=0; i<ncol; ++i) {
      for (int k=0; k<nlev; ++k) {
        REQUIRE (rate(i,0,k) == 100 + 3*300);
      }
    }
  }

  SECTION ("tensors") {
    // Inputs and outputs are named, (ncol, nlev), row-major, in the configured order
    ekat::ParameterList p("emu");
    p.set<std::vector<std::string>>("inputs", {"x", "b"});
    p.set<std::vector<std::string>>("outputs", {"c"});
    auto backend = std::make_shared<FunctionBackend>([&](const TensorMap& in, TensorMap& out) {
      std::vector<std::string> names;
      for (const auto& t : in) {
        names.push_back(t.name());
        REQUIRE (t.dims()==std::vector<std::int64_t>{ncol, nlev});
        REQUIRE (not t.writable());
      }
      REQUIRE (names==std::vector<std::string>{"x", "b"});
      const double* xi = find(in, "x");
      REQUIRE (xi[1*nlev + 2] == Approx(1 + 0.2));
      double* c = find(out, "c");
      for (int n=0; n<ncol*nlev; ++n) c[n] = 0;
    });
    PE emu("emu", p, rate_names, ncol, nlev, backend);
    reset();
    emu.run(state, rates, rates);
    REQUIRE (rate(0,2,0) == 0);
  }

  SECTION ("errors") {
    auto backend = std::make_shared<FunctionBackend>([](const TensorMap&, TensorMap&) {});
    ekat::ParameterList p("emu");
    p.set<std::vector<std::string>>("inputs", {"x"});

    // An output that is neither a rate nor a mask
    p.set<std::vector<std::string>>("outputs", {"a", "z"});
    REQUIRE_THROWS (PE("emu", p, rate_names, ncol, nlev, backend));

    // A mask gating something that is not an output
    p.set<std::vector<std::string>>("outputs", {"a", "m"});
    p.sublist("masks").set<std::vector<std::string>>("m", {"b"});
    REQUIRE_THROWS (PE("emu", p, rate_names, ncol, nlev, backend));

    // An input that does not exist
    ekat::ParameterList q("emu");
    q.set<std::vector<std::string>>("inputs", {"y"});
    q.set<std::vector<std::string>>("outputs", {"a"});
    PE emu("emu", q, rate_names, ncol, nlev, backend);
    reset();
    REQUIRE_THROWS (emu.run(state, rates, rates));
  }
}

} // namespace scream
