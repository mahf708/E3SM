// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "atm.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace emulator {
namespace test {

namespace {

constexpr int NCOL = 6;
constexpr int NX = 3;
constexpr int NY = 2;

/// Write a namelist and remove it when the test ends.
class ScopedFile {
public:
  ScopedFile(std::string path, const std::string &contents)
      : m_path(std::move(path)) {
    std::ofstream ofs(m_path);
    ofs << contents;
  }
  ~ScopedFile() { std::remove(m_path.c_str()); }
  const std::string &path() const { return m_path; }

private:
  std::string m_path;
};

/// The decomposition the coupler would hand a single-rank component.
struct FakeGrid {
  std::vector<int> gids;
  std::vector<double> lat, lon, area;

  FakeGrid() {
    for (int c = 0; c < NCOL; ++c) {
      gids.push_back(c + 1);
      lat.push_back(static_cast<double>(c / NX));
      lon.push_back(static_cast<double>(c % NX));
      area.push_back(1.0);
    }
  }

  EmulatorGridDesc desc() const {
    EmulatorGridDesc grid{};
    grid.grid_type = 0;
    grid.nx = NX;
    grid.ny = NY;
    grid.num_local_cols = NCOL;
    grid.num_global_cols = NCOL;
    grid.col_gids = gids.data();
    grid.lat = lat.data();
    grid.lon = lon.data();
    grid.area = area.data();
    return grid;
  }
};

/// Wire up a component the way the MCT layer does.
///
/// No nx/ny in the namelist, so create_instance does not try to build its own
/// decomposition (which would need a live MPI communicator); the coupler
/// supplies one through set_grid_data, as it does in a real run.
void configure(EmulatorAtm &atm, const FakeGrid &grid,
               const ScopedFile &namelist, const std::string &export_fields,
               const std::string &import_fields, double *x2a, double *a2x,
               int n_import, int n_export, int field_size) {
  atm.create_instance(0, 1, namelist.path(), "", 0, 20260101, 0);
  const auto desc = grid.desc();
  atm.set_grid_data(desc);
  atm.init_coupling_indices(export_fields, import_fields);

  EmulatorCouplingDesc cpl{};
  cpl.import_data = x2a;
  cpl.export_data = a2x;
  cpl.num_imports = n_import;
  cpl.num_exports = n_export;
  cpl.field_size = field_size;
  atm.setup_coupling(cpl);
}

} // namespace

TEST_CASE("A field goes from x2a through the model and back into a2x",
          "[atm][coupling]") {
  // The fixture emulator computes y = scale * x + step.
  ScopedFile namelist("emulator_atm_test_in",
                      "inference.backend: python\n"
                      "inference.python_module: emulator_fixture\n"
                      "inference.python_path: " EMULATOR_TEST_FIXTURE_DIR "\n"
                      "inference.report_path: emulator_atm_test_report.txt\n"
                      "inference.scale: 2.0\n"
                      "inference.input: x\n"
                      "inference.output: y\n");

  // x2a and a2x are Fortran rAttr(nflds, lsize): field-contiguous, column
  // strided.  Lay them out exactly that way.
  const int n_import = 4;
  const int n_export = 3;
  std::vector<double> x2a(static_cast<std::size_t>(n_import) * NCOL, -7.0);
  std::vector<double> a2x(static_cast<std::size_t>(n_export) * NCOL, -7.0);
  for (int c = 0; c < NCOL; ++c) {
    x2a[static_cast<std::size_t>(c) * n_import + 2] = c + 1.0; // row of "x"
  }

  EmulatorAtm atm;
  const FakeGrid grid;
  configure(atm, grid, namelist, /*export=*/"a:y:b", /*import=*/"p:q:x:r",
            x2a.data(), a2x.data(), n_import, n_export, NCOL);

  atm.initialize();
  atm.run(1800);

  // y = 2 * x + 1 on the first step, written into the "y" row of a2x.
  for (int c = 0; c < NCOL; ++c) {
    const double y = a2x[static_cast<std::size_t>(c) * n_export + 1];
    REQUIRE(y == Approx(2.0 * (c + 1.0) + 1.0));
  }

  // The rows the model does not touch must be exactly as the coupler left
  // them: a gather or scatter that strides wrongly would smear across them.
  for (int c = 0; c < NCOL; ++c) {
    REQUIRE(a2x[static_cast<std::size_t>(c) * n_export + 0] == -7.0);
    REQUIRE(a2x[static_cast<std::size_t>(c) * n_export + 2] == -7.0);
  }

  // A second step proves the model object, and its state, persist.
  atm.run(1800);
  for (int c = 0; c < NCOL; ++c) {
    const double y = a2x[static_cast<std::size_t>(c) * n_export + 1];
    REQUIRE(y == Approx(2.0 * (c + 1.0) + 2.0));
  }

  atm.finalize();
  std::remove("emulator_atm_test_report.txt");
}

TEST_CASE("An input the coupler does not carry is fatal", "[atm][coupling]") {
  // An unmatched input has no other source, so allowing one is permission to
  // run the model on zeros.
  ScopedFile namelist("emulator_atm_test_missing_in",
                      "inference.backend: python\n"
                      "inference.python_module: emulator_fixture\n"
                      "inference.python_path: " EMULATOR_TEST_FIXTURE_DIR "\n"
                      "inference.input: not_a_coupling_field\n"
                      "inference.output: b\n");

  EmulatorAtm atm;
  const FakeGrid grid;
  std::vector<double> x2a(2 * NCOL, 1.0);
  std::vector<double> a2x(2 * NCOL, 5.0);
  configure(atm, grid, namelist, "a:b", "p:q", x2a.data(), a2x.data(), 2, 2,
            NCOL);

  REQUIRE_THROWS_WITH(atm.initialize(), Catch::Contains("not_a_coupling_field") &&
                                            Catch::Contains("run on zeros"));
}

TEST_CASE("A zero-filled input can be allowed deliberately",
          "[atm][coupling]") {
  ScopedFile namelist("emulator_atm_test_allowed_in",
                      "inference.backend: stub\n"
                      "inference.unsafe_allow_zero_filled_inputs: true\n"
                      "inference.input: supplied_elsewhere\n"
                      "inference.output: also_not_a_field\n");

  EmulatorAtm atm;
  const FakeGrid grid;
  std::vector<double> x2a(2 * NCOL, 1.0);
  std::vector<double> a2x(2 * NCOL, 5.0);
  configure(atm, grid, namelist, "a:b", "p:q", x2a.data(), a2x.data(), 2, 2,
            NCOL);

  // An unmatched *output* stays non-fatal: a model may produce diagnostics
  // the coupler does not consume.
  atm.initialize();
  atm.run(1800);
  for (double value : a2x) {
    REQUIRE(value == 5.0);
  }
  atm.finalize();
}

TEST_CASE("A column-count mismatch is fatal", "[atm][coupling]") {
  // Truncating to the shorter of the two would leave part of every field
  // unset -- plausible numbers over part of the globe, zeros over the rest.
  ScopedFile namelist("emulator_atm_test_size_in", "inference.input: p\n"
                                                   "inference.output: a\n");

  EmulatorAtm atm;
  const FakeGrid grid;
  std::vector<double> x2a(2 * NCOL, 1.0);
  std::vector<double> a2x(2 * NCOL, 5.0);
  configure(atm, grid, namelist, "a:b", "p:q", x2a.data(), a2x.data(), 2, 2,
            NCOL - 1);

  REQUIRE_THROWS_WITH(atm.initialize(), Catch::Contains("disagree"));
}

TEST_CASE("A field list that disagrees with the buffer is fatal",
          "[atm][coupling]") {
  // If the list and the attribute vector describe different things, every
  // resolved row index is off by an unknown amount.
  ScopedFile namelist("emulator_atm_test_list_in", "inference.input: p\n"
                                                   "inference.output: a\n");

  EmulatorAtm atm;
  const FakeGrid grid;
  std::vector<double> x2a(3 * NCOL, 1.0);
  std::vector<double> a2x(2 * NCOL, 5.0);
  configure(atm, grid, namelist, "a:b", "p:q", x2a.data(), a2x.data(), 3, 2,
            NCOL);

  REQUIRE_THROWS_WITH(atm.initialize(), Catch::Contains("x2a holds 3"));
}

TEST_CASE("No coupling buffers at all is fine for the stub",
          "[atm][coupling]") {
  // The stub runs no model, so unfed inputs are exactly what it is for.
  ScopedFile namelist("emulator_atm_test_nocpl_in", "inference.backend: stub\n"
                                                    "inference.input: p\n"
                                                    "inference.output: a\n");

  EmulatorAtm atm;
  const FakeGrid grid;
  atm.create_instance(0, 1, namelist.path(), "", 0, 20260101, 0);
  const auto desc = grid.desc();
  atm.set_grid_data(desc);

  atm.initialize();
  atm.run(1800);
  atm.finalize();
}

TEST_CASE("No coupling buffers is fatal for a real backend",
          "[atm][coupling]") {
  // A python or ACE backend with declared inputs and nothing feeding them
  // would run on zeros and return plausible numbers.
  ScopedFile namelist("emulator_atm_test_nocpl_real_in",
                      "inference.backend: python\n"
                      "inference.python_module: emulator_fixture\n"
                      "inference.python_path: " EMULATOR_TEST_FIXTURE_DIR "\n"
                      "inference.input: x\n"
                      "inference.output: y\n");

  EmulatorAtm atm;
  const FakeGrid grid;
  atm.create_instance(0, 1, namelist.path(), "", 0, 20260101, 0);
  const auto desc = grid.desc();
  atm.set_grid_data(desc);

  REQUIRE_THROWS_WITH(atm.initialize(), Catch::Contains("run on zeros"));
}

} // namespace test
} // namespace emulator
