// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "horizontal_grid.hpp"

#include <numbers>
#include <set>
#include <vector>

namespace emulator {
namespace grid {
namespace test {

/**
 * A 4 x 2 regular grid covering the sphere exactly: two latitude bands,
 * [-90, 0] and [0, 90], four quarter-longitudes.  Each cell's solid angle is
 * (pi/2) * (sin(top) - sin(bottom)) = pi/2, so the eight sum to 4 pi.
 */
HorizontalGrid tiny_global_grid() {
  HorizontalGrid g;
  g.name = "tiny";
  g.nx = 4;
  g.ny = 2;
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 4; ++i) {
      g.lat.push_back(j == 0 ? -45.0 : 45.0);
      g.lon.push_back(45.0 + 90.0 * i);
      g.area.push_back(std::numbers::pi / 2.0);
      g.imask.push_back(1);
    }
  }
  return g;
}

TEST_CASE("A consistent global grid validates", "[grid]") {
  const auto g = tiny_global_grid();
  REQUIRE_NOTHROW(g.validate(true));
  REQUIRE(g.total_area() == Approx(4.0 * std::numbers::pi));
}

TEST_CASE("Grid validation names what is wrong", "[grid]") {
  auto g = tiny_global_grid();

  SECTION("a short array") {
    g.area.pop_back();
    REQUIRE_THROWS_WITH(g.validate(), Catch::Contains("area has 7 entries") &&
                                          Catch::Contains("nx*ny is 8"));
  }
  SECTION("latitudes in radians labelled as degrees would pass, so the "
          "range check is only a backstop; a value past 90 is caught") {
    g.lat[3] = 91.0;
    REQUIRE_THROWS_WITH(g.validate(), Catch::Contains("outside [-90, 90]") &&
                                          Catch::Contains("cell 3"));
  }
  SECTION("a zero area") {
    g.area[5] = 0.0;
    REQUIRE_THROWS_WITH(g.validate(), Catch::Contains("cell 5"));
  }
  SECTION("a non-binary imask") {
    g.imask[0] = 2;
    REQUIRE_THROWS_WITH(g.validate(), Catch::Contains("not 0 or 1"));
  }
  SECTION("areas in square degrees") {
    const double sq_deg_per_sr = (180.0 / std::numbers::pi) *
                                 (180.0 / std::numbers::pi);
    for (auto &a : g.area) {
      a *= sq_deg_per_sr;
    }
    REQUIRE_NOTHROW(g.validate(false));
    REQUIRE_THROWS_WITH(g.validate(true), Catch::Contains("square degrees"));
  }
  SECTION("a lost row") {
    g.ny = 1;
    g.lat.resize(4);
    g.lon.resize(4);
    g.area.resize(4);
    g.imask.resize(4);
    REQUIRE_NOTHROW(g.validate(false));
    REQUIRE_THROWS_WITH(g.validate(true), Catch::Contains("4 pi"));
  }
}

TEST_CASE("Contiguous blocks cover every cell exactly once", "[grid]") {
  // Including more ranks than cells, which leaves some ranks empty: normal on
  // a small grid with a large layout, and it must not break anything.
  for (const std::size_t ncells : {1u, 7u, 8u, 64800u}) {
    for (const int nranks : {1, 2, 3, 7, 8, 16}) {
      std::set<int> seen;
      std::size_t smallest = ncells;
      std::size_t largest = 0;
      std::size_t next_offset = 0;
      for (int r = 0; r < nranks; ++r) {
        const auto d = Decomposition::contiguous_blocks(ncells, nranks, r);
        REQUIRE(d.offset() == next_offset);
        next_offset += d.num_local();
        smallest = std::min(smallest, d.num_local());
        largest = std::max(largest, d.num_local());
        for (const int id : d.global_ids()) {
          REQUIRE(seen.insert(id).second);
        }
      }
      REQUIRE(seen.size() == ncells);
      REQUIRE(*seen.begin() == 1);
      REQUIRE(largest - smallest <= 1);
    }
  }
}

TEST_CASE("A decomposition slices global arrays and refuses wrong sizes",
          "[grid]") {
  const auto d = Decomposition::contiguous_blocks(8, 3, 1); // cells 3,4,5
  REQUIRE(d.offset() == 3);
  REQUIRE(d.num_local() == 3);
  REQUIRE(d.global_ids() == std::vector<int>{4, 5, 6});

  const std::vector<double> global{0, 1, 2, 3, 4, 5, 6, 7};
  REQUIRE(d.local(global) == std::vector<double>{3, 4, 5});

  const std::vector<double> wrong(9, 0.0);
  REQUIRE_THROWS_WITH(d.local(wrong), Catch::Contains("9 values") &&
                                          Catch::Contains("8 cells"));
  REQUIRE_THROWS_AS(Decomposition::contiguous_blocks(8, 0, 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(Decomposition::contiguous_blocks(8, 2, 2),
                    std::invalid_argument);
}

TEST_CASE("A full domain has mask and frac of one, and real coordinates",
          "[grid]") {
  const auto g = tiny_global_grid();
  const auto d = Decomposition::contiguous_blocks(g.size(), 2, 1);
  const auto dom = Domain::full(g, d);
  REQUIRE(dom.size() == 4);
  REQUIRE(dom.global_ids == std::vector<int>{5, 6, 7, 8});
  // Guards against latitude coming back as 0 everywhere instead of the
  // grid's real coordinates.
  REQUIRE(dom.lat == std::vector<double>{45, 45, 45, 45});
  REQUIRE(dom.lon[3] == 315.0);
  REQUIRE(dom.mask == std::vector<double>(4, 1.0));
  REQUIRE(dom.frac == dom.mask);
}

TEST_CASE("A surface domain takes a binary mask and sets frac equal to it",
          "[grid]") {
  const auto g = tiny_global_grid();
  const auto d = Decomposition::contiguous_blocks(g.size(), 1, 0);
  const std::vector<double> ocean{1, 1, 0, 1, 0, 0, 1, 1};

  const auto dom = Domain::masked(g, d, ocean);
  REQUIRE(dom.mask == ocean);
  REQUIRE(dom.frac == ocean);
}

TEST_CASE("A continuous fraction is refused as a mask", "[grid]") {
  // A sea surface fraction of 0.37 on a coastal cell: fine as model data,
  // wrong as the coupler's mask.
  const auto g = tiny_global_grid();
  const auto d = Decomposition::contiguous_blocks(g.size(), 1, 0);
  const std::vector<double> sea_fraction{1, 1, 0.37, 1, 0, 0, 0.5, 1};
  REQUIRE_THROWS_WITH(Domain::masked(g, d, sea_fraction),
                      Catch::Contains("2 cells") &&
                          Catch::Contains("cell 2, value 0.37") &&
                          Catch::Contains("coastal"));

  const std::vector<double> short_mask(7, 1.0);
  REQUIRE_THROWS_WITH(Domain::masked(g, d, short_mask),
                      Catch::Contains("7 values for 8 cells"));
}

} // namespace test
} // namespace grid
} // namespace emulator
