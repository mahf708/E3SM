// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "emulator.hpp"
#include "emulator_registry.hpp"
#include "emulator_c_api.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace emulator {
namespace test {

// Concrete implementation for testing
class TestEmulator : public Emulator {
public:
  TestEmulator(EmulatorType type = EmulatorType::ATM_COMP, 
               int id = -1, const std::string &name = "")
      : Emulator(type, id, name) {}

  // Track calls for verification
  bool init_called = false;
  bool run_called = false;
  bool final_called = false;
  int last_dt = 0;

   // Stub implementations of the pure virtuals from Emulator
  void set_grid_data(const EmulatorGridDesc&) override {}

  int get_num_local_cols()  const override { return 0; }
  int get_num_global_cols() const override { return 0; }
  int get_nx()              const override { return 0; }
  int get_ny()              const override { return 0; }

  void get_local_col_gids(int* ) const override {}
  void get_cols_latlon(double*, double* ) const override {}
  void get_cols_area(double* ) const override {}

protected:
  void init_impl() override { init_called = true; }
  void run_impl(int dt) override {
    run_called = true;
    last_dt = dt;
  }
  void final_impl() override { final_called = true; }
};

// Test emulators for different EmulatorTypes
class TestOcnEmulator : public Emulator {
public:
  TestOcnEmulator(int id = -1, const std::string &name = "")
      : Emulator(EmulatorType::OCN_COMP, id, name) {}

protected:
  void init_impl() override {}
  void run_impl(int) override {}
  void final_impl() override {}
};

class TestIceEmulator : public Emulator {
public:
  TestIceEmulator(int id = -1, const std::string &name = "")
      : Emulator(EmulatorType::ICE_COMP, id, name) {}

protected:
  void init_impl() override {}
  void run_impl(int) override {}
  void final_impl() override {}
};

class TestLndEmulator : public Emulator {
public:
  TestLndEmulator(int id = -1, const std::string &name = "")
      : Emulator(EmulatorType::LND_COMP, id, name) {}

protected:
  void init_impl() override {}
  void run_impl(int) override {}
  void final_impl() override {}
};

TEST_CASE("Emulator construction", "[emulator]") {
  TestEmulator emu;
  REQUIRE(emu.type() == EmulatorType::ATM_COMP);
  REQUIRE(emu.id() == -1);
  REQUIRE(emu.name().empty());
  REQUIRE_FALSE(emu.is_initialized());
  REQUIRE(emu.step_count() == 0);
}

TEST_CASE("Emulator construction with args", "[emulator]") {
  TestEmulator emu(EmulatorType::ATM_COMP, 42, "test_atm");
  REQUIRE(emu.id() == 42);
  REQUIRE(emu.name() == "test_atm");
}

TEST_CASE("Emulator different types", "[emulator]") {

  TestEmulator atm(EmulatorType::ATM_COMP);
  TestEmulator ocn(EmulatorType::OCN_COMP);
  TestEmulator ice(EmulatorType::ICE_COMP);
  TestEmulator lnd(EmulatorType::LND_COMP);

  REQUIRE(atm.type() == EmulatorType::ATM_COMP);
  REQUIRE(ocn.type() == EmulatorType::OCN_COMP);
  REQUIRE(ice.type() == EmulatorType::ICE_COMP);
  REQUIRE(lnd.type() == EmulatorType::LND_COMP);
}

TEST_CASE("Emulator lifecycle", "[emulator]") {
  TestEmulator emu(EmulatorType::ATM_COMP, 1, "test");

  SECTION("initialize calls init_impl") {
    REQUIRE_FALSE(emu.init_called);
    emu.initialize();
    REQUIRE(emu.init_called);
    REQUIRE(emu.is_initialized());
  }

  SECTION("run calls run_impl and increments step count") {
    emu.initialize();
    REQUIRE(emu.step_count() == 0);

    emu.run(3600);
    REQUIRE(emu.run_called);
    REQUIRE(emu.last_dt == 3600);
    REQUIRE(emu.step_count() == 1);

    emu.run(1800);
    REQUIRE(emu.step_count() == 2);
  }

  SECTION("finalize calls final_impl") {
    emu.initialize();
    REQUIRE_FALSE(emu.final_called);
    emu.finalize();
    REQUIRE(emu.final_called);
    REQUIRE_FALSE(emu.is_initialized());
  }
}

TEST_CASE("Emulator error handling", "[emulator]") {
  TestEmulator emu(EmulatorType::ATM_COMP,1, "test");

  SECTION("run before initialize throws") {
    REQUIRE_THROWS_AS(emu.run(100), std::runtime_error);
  }

  SECTION("double initialize throws") {
    emu.initialize();
    REQUIRE_THROWS_AS(emu.initialize(), std::runtime_error);
  }

  SECTION("finalize without initialize is safe") {
    REQUIRE_NOTHROW(emu.finalize());
  }

  SECTION("re-initialization after finalize works") {
    emu.initialize();
    REQUIRE(emu.is_initialized());
    emu.run(100);
    REQUIRE(emu.step_count() == 1);

    emu.finalize();
    REQUIRE_FALSE(emu.is_initialized());

    emu.initialize();
    REQUIRE(emu.is_initialized());
    emu.run(200);
    REQUIRE(emu.step_count() == 2); // step count persists across finalize/re-init
  }
}

TEST_CASE("Emulator with EmulatorRegistry", "[emulator][integration]") {
  auto &reg = EmulatorRegistry::instance();
  reg.clean_up();

  auto &emu = reg.create<TestEmulator>("test_emu", EmulatorType::ATM_COMP, 99, "registry_test");

  REQUIRE(reg.has("test_emu"));
  REQUIRE(emu.type() == EmulatorType::ATM_COMP);
  REQUIRE(emu.id() == 99);

  emu.initialize();
  emu.run(100);

  const auto &ref = reg.get<TestEmulator>("test_emu");
  REQUIRE(ref.id() == 99);
  REQUIRE(ref.step_count() == 1);

  auto &mut_ref = reg.get_mut<TestEmulator>("test_emu");
  mut_ref.run(200);
  REQUIRE(ref.step_count() == 2);

  reg.clean_up();
}

// ---------------------------------------------------------------------------
// Coupling through the base class
// ---------------------------------------------------------------------------

/// A three-column ocean: reads a heat flux, exports SST as 270 + flux/100.
class CoupledOcn : public Emulator {
public:
  CoupledOcn() : Emulator(EmulatorType::OCN_COMP, 1, "coupled_ocn") {}

  int seen_steps = 0;
  double first_flux_seen = -1.0;

  void set_grid_data(const EmulatorGridDesc &) override {}
  int get_num_local_cols() const override { return 3; }
  int get_num_global_cols() const override { return 3; }
  int get_nx() const override { return 3; }
  int get_ny() const override { return 1; }
  void get_local_col_gids(int *) const override {}
  void get_cols_latlon(double *, double *) const override {}
  void get_cols_area(double *) const override {}

protected:
  CouplingFields coupling_fields() const override {
    return {{{"Foxx_swnet"}, {"Foxx_rofl", fields::Need::Optional}},
            {{"So_t"}}};
  }
  void init_impl() override {
    auto sst = mutable_exports().get("So_t");
    std::fill(sst.begin(), sst.end(), 271.0);
  }
  void run_impl(int) override {
    const auto flux = imports().get("Foxx_swnet");
    auto sst = mutable_exports().get("So_t");
    if (seen_steps == 0) {
      first_flux_seen = flux[0];
    }
    for (std::size_t i = 0; i < sst.size(); ++i) {
      sst[i] = 270.0 + flux[i] / 100.0;
    }
    ++seen_steps;
  }
  void final_impl() override {}
};

/// MCT's point-major layout: field f at point p is data[p*nfields + f].
struct AttrVect {
  std::size_t nfields;
  std::size_t npoints;
  std::vector<double> data;
  AttrVect(std::size_t nf, std::size_t np)
      : nfields(nf), npoints(np), data(nf * np, -999.0) {}
  double &at(std::size_t f, std::size_t p) { return data[p * nfields + f]; }
};

TEST_CASE("The base class pulls before run and pushes after init and run",
          "[emulator][coupling]") {
  CoupledOcn ocn;
  AttrVect x2o(3, 3); // Foxx_taux:Foxx_swnet:Sa_pslv
  AttrVect o2x(2, 3); // So_u:So_t

  ocn.set_coupler_field_lists("Foxx_taux:Foxx_swnet:Sa_pslv", "So_u:So_t");
  EmulatorCouplingDesc cpl{x2o.data.data(), o2x.data.data(), 3, 2, 3};
  ocn.setup_coupling(cpl);
  REQUIRE(ocn.is_coupled());
  REQUIRE(ocn.import_binding()->absent() ==
          std::vector<std::string>{"Foxx_rofl"});
  REQUIRE(ocn.export_binding()->unbound() == std::vector<std::string>{"So_u"});

  ocn.initialize();
  // The initial state reached the coupler before any run, and the field the
  // ocean does not produce was zeroed rather than left at -999.
  for (std::size_t p = 0; p < 3; ++p) {
    REQUIRE(o2x.at(1, p) == 271.0);
    REQUIRE(o2x.at(0, p) == 0.0);
  }

  // The coupler writes this step's fluxes, then calls run.
  for (std::size_t p = 0; p < 3; ++p) {
    x2o.at(1, p) = 100.0 * static_cast<double>(p + 1);
    x2o.at(0, p) = 5.0; // taux, which the ocean does not read
  }
  ocn.run(1800);
  REQUIRE(ocn.first_flux_seen == 100.0);
  REQUIRE(o2x.at(1, 0) == 271.0);
  REQUIRE(o2x.at(1, 2) == 273.0);
  REQUIRE(ocn.step_count() == 1);
}

TEST_CASE("Coupling setup refuses what would otherwise run wrong",
          "[emulator][coupling]") {
  CoupledOcn ocn;
  AttrVect x2o(2, 3);
  AttrVect o2x(1, 3);

  SECTION("buffers before names") {
    EmulatorCouplingDesc cpl{x2o.data.data(), o2x.data.data(), 2, 1, 3};
    REQUIRE_THROWS_WITH(ocn.setup_coupling(cpl),
                        Catch::Contains("set_coupler_field_lists"));
  }

  SECTION("import and export lists passed in the wrong order") {
    // Import and export lists swapped: the component asks for Foxx_swnet
    // among the export names, and says so.
    ocn.set_coupler_field_lists("So_t", "Foxx_swnet:Sa_pslv");
    EmulatorCouplingDesc cpl{o2x.data.data(), x2o.data.data(), 1, 2, 3};
    REQUIRE_THROWS_WITH(ocn.setup_coupling(cpl),
                        Catch::Contains("Foxx_swnet"));
  }

  SECTION("a decomposition that disagrees with the component") {
    ocn.set_coupler_field_lists("Foxx_swnet:Sa_pslv", "So_t");
    EmulatorCouplingDesc cpl{x2o.data.data(), o2x.data.data(), 2, 1, 4};
    REQUIRE_THROWS_WITH(ocn.setup_coupling(cpl),
                        Catch::Contains("4 points") &&
                            Catch::Contains("owns 3"));
  }

  SECTION("a truncated field list") {
    ocn.set_coupler_field_lists("Foxx_swnet", "So_t");
    EmulatorCouplingDesc cpl{x2o.data.data(), o2x.data.data(), 2, 1, 3};
    REQUIRE_THROWS_WITH(ocn.setup_coupling(cpl),
                        Catch::Contains("truncated"));
  }

  REQUIRE_FALSE(ocn.is_coupled());
}

TEST_CASE("An uncoupled emulator still runs", "[emulator][coupling]") {
  // Standalone use -- a unit test, an offline driver -- never calls the
  // coupling setup, and nothing is pulled or pushed.
  TestEmulator emu(EmulatorType::ATM_COMP, 1, "uncoupled");
  REQUIRE_FALSE(emu.is_coupled());
  emu.initialize();
  REQUIRE_NOTHROW(emu.run(3600));
  REQUIRE(emu.imports().size() == 0);
}

// ---------------------------------------------------------------------------
// Grid through the base class
// ---------------------------------------------------------------------------

/// Nothing but the lifecycle: every grid getter is the base class's.
class GridOnly : public Emulator {
public:
  GridOnly() : Emulator(EmulatorType::ICE_COMP, 1, "grid_only") {}

protected:
  void init_impl() override {}
  void run_impl(int) override {}
  void final_impl() override {}
};

TEST_CASE("The base class keeps the grid it is given", "[emulator][grid]") {
  GridOnly emu;
  REQUIRE_FALSE(emu.has_domain());
  REQUIRE(emu.get_num_local_cols() == 0);

  const int gids[] = {3, 4};
  const double lat[] = {-89.2, -89.2};
  const double lon[] = {0.5, 1.5};
  const double area[] = {1e-5, 1e-5};
  EmulatorGridDesc desc{0, 360, 180, 2, 64800, gids, lat, lon, area};
  emu.set_grid_data(desc);

  REQUIRE(emu.has_domain());
  REQUIRE(emu.get_num_local_cols() == 2);
  REQUIRE(emu.get_num_global_cols() == 64800);
  REQUIRE(emu.get_nx() == 360);

  std::vector<int> got_ids(2);
  std::vector<double> got_lat(2), got_lon(2), mask(2), frac(2);
  emu.get_local_col_gids(got_ids.data());
  emu.get_cols_latlon(got_lat.data(), got_lon.data());
  emu.get_cols_mask_frac(mask.data(), frac.data());
  REQUIRE(got_ids == std::vector<int>{3, 4});
  REQUIRE(got_lat[1] == -89.2);
  REQUIRE(got_lon[1] == 1.5);
  REQUIRE(mask == std::vector<double>{1.0, 1.0});
  REQUIRE(frac == mask);
}

TEST_CASE("The grid cannot change once coupling is set up",
          "[emulator][grid]") {
  GridOnly emu;
  const int gids[] = {1, 2};
  const double lat[] = {0.0, 0.0};
  const double lon[] = {0.0, 1.0};
  const double area[] = {1.0, 1.0};
  EmulatorGridDesc desc{0, 2, 1, 2, 2, gids, lat, lon, area};
  emu.set_grid_data(desc);

  std::vector<double> x2i(2 * 1, 0.0), i2x(2 * 1, 0.0);
  emu.set_coupler_field_lists("Sa_z", "Si_t");
  EmulatorCouplingDesc cpl{x2i.data(), i2x.data(), 1, 1, 2};
  emu.setup_coupling(cpl);

  REQUIRE_THROWS_WITH(emu.set_grid_data(desc),
                      Catch::Contains("after setup_coupling"));
}

TEST_CASE("A grid descriptor with columns and no coordinates is refused",
          "[emulator][grid]") {
  GridOnly emu;
  const int gids[] = {1};
  EmulatorGridDesc desc{0, 1, 1, 1, 1, gids, nullptr, nullptr, nullptr};
  REQUIRE_THROWS_WITH(emu.set_grid_data(desc),
                      Catch::Contains("null coordinate"));
  REQUIRE_FALSE(emu.has_domain());
}

TEST_CASE("EmulatorRegistry get_mut throws for unknown name",
          "[emulator_registry]") {
  auto &reg = EmulatorRegistry::instance();
  reg.clean_up();

  REQUIRE_THROWS_AS(reg.get_mut<TestEmulator>("nonexistent"),
                    std::runtime_error);

  reg.clean_up();
}

} // namespace test
} // namespace emulator
