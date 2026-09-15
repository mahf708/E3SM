// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "coupler_binding.hpp"
#include "field_list.hpp"
#include "field_set.hpp"
#include "mask_set.hpp"

#include <string>
#include <vector>

namespace emulator {
namespace fields {
namespace test {

namespace {

/**
 * A buffer laid out the way Fortran lays out `rAttr(nfields, npoints)`:
 * column-major, so C sees point-major.  Field f at point p holds 100*f + p,
 * which makes a stride mistake visible in any single value.
 */
std::vector<double> fortran_attr_vect(std::size_t nfields,
                                      std::size_t npoints) {
  std::vector<double> data(nfields * npoints);
  for (std::size_t p = 0; p < npoints; ++p) {
    for (std::size_t f = 0; f < nfields; ++f) {
      data[p * nfields + f] = 100.0 * static_cast<double>(f) +
                              static_cast<double>(p);
    }
  }
  return data;
}

} // namespace

// ---------------------------------------------------------------------------
// FieldList
// ---------------------------------------------------------------------------

TEST_CASE("FieldList parses the coupler's colon-separated form", "[fields]") {
  const auto list = FieldList::parse("Sa_z:Sa_u: Sa_v :Sa_tbot");
  REQUIRE(list.size() == 4);
  REQUIRE(list.name(2) == "Sa_v");
  REQUIRE(list.find("Sa_tbot") == 3u);
  REQUIRE_FALSE(list.contains("Sa_ptem"));
  REQUIRE(list.to_string() == "Sa_z:Sa_u:Sa_v:Sa_tbot");
}

TEST_CASE("FieldList stops at a NUL, as a Fortran buffer arrives",
          "[fields]") {
  const std::string buffer("So_t:So_s\0\0\0garbage", 20);
  const auto list = FieldList::parse(buffer);
  REQUIRE(list.size() == 2);
  REQUIRE(list.name(1) == "So_s");
}

TEST_CASE("FieldList takes an empty string as an empty list", "[fields]") {
  REQUIRE(FieldList::parse("").empty());
  REQUIRE(FieldList::parse("   ").empty());
}

TEST_CASE("FieldList refuses empty and duplicated names", "[fields]") {
  REQUIRE_THROWS_WITH(FieldList::parse("Sa_z::Sa_u"),
                      Catch::Contains("entry 2 is empty"));
  REQUIRE_THROWS_WITH(FieldList::parse("Sa_z:Sa_u:Sa_z"),
                      Catch::Contains("'Sa_z' appears twice") &&
                          Catch::Contains("at 1 and 3"));
}

TEST_CASE("FieldList keeps every name of a long list", "[fields]") {
  // seq_flds_x2o_fields runs well past 256 characters; FieldList must have
  // no length limit of its own.
  std::string joined;
  for (int i = 0; i < 300; ++i) {
    joined += (i ? ":" : "") + std::string("Foxx_field_") + std::to_string(i);
  }
  REQUIRE(joined.size() > 4000);
  const auto list = FieldList::parse(joined);
  REQUIRE(list.size() == 300);
  REQUIRE(list.name(299) == "Foxx_field_299");
}

TEST_CASE("FieldList names near misses by case", "[fields]") {
  const auto list = FieldList::parse("So_t:So_s:Si_ifrac");
  REQUIRE(list.near_misses("So_T") == std::vector<std::string>{"So_t"});
  REQUIRE(list.near_misses("So_t").empty());
  REQUIRE(list.near_misses("Sx_t").empty());
}

// ---------------------------------------------------------------------------
// FieldSet
// ---------------------------------------------------------------------------

TEST_CASE("FieldSet holds contiguous named fields", "[fields]") {
  FieldSet set(5);
  auto t = set.add("So_t", 273.15);
  REQUIRE(t.size() == 5);
  REQUIRE(t[4] == 273.15);

  t[2] = 300.0;
  REQUIRE(set.get("So_t")[2] == 300.0);
  REQUIRE(set.names() == std::vector<std::string>{"So_t"});

  REQUIRE_THROWS_WITH(set.add("So_t"), Catch::Contains("already"));
  REQUIRE_THROWS_WITH(set.add(""), Catch::Contains("needs a name"));
}

TEST_CASE("A FieldSet span survives adding more fields", "[fields]") {
  FieldSet set(4);
  auto first = set.add("a");
  const double *before = first.data();
  for (int i = 0; i < 64; ++i) {
    set.add("f" + std::to_string(i));
  }
  first[3] = 7.0;
  REQUIRE(set.get("a").data() == before);
  REQUIRE(set.get("a")[3] == 7.0);
}

TEST_CASE("A missing FieldSet field is reported with what is there",
          "[fields]") {
  FieldSet set(1);
  set.add("So_t");
  set.add("So_s");
  REQUIRE_THROWS_WITH(set.get("So_u"), Catch::Contains("'So_u'") &&
                                           Catch::Contains("So_t, So_s"));
}

// ---------------------------------------------------------------------------
// AttrVectView
// ---------------------------------------------------------------------------

TEST_CASE("AttrVectView reads MCT's point-major layout", "[fields]") {
  // 3 fields on 5 points: nfields != npoints, so a transposed stride
  // produces a different value, not the same one by symmetry.
  const auto layout = FieldList::parse("a:b:c");
  auto data = fortran_attr_vect(3, 5);
  AttrVectView av(data.data(), layout, 3, 5);

  REQUIRE(av.at(1, 0) == 100.0);
  REQUIRE(av.at(2, 4) == 204.0);
  REQUIRE(av.at(0, 3) == 3.0);

  std::vector<double> b(5);
  av.read(1, b);
  REQUIRE(b == std::vector<double>{100, 101, 102, 103, 104});

  const std::vector<double> c{-1, -2, -3, -4, -5};
  av.write(2, c);
  REQUIRE(data[0 * 3 + 2] == -1.0);
  REQUIRE(data[4 * 3 + 2] == -5.0);
  // and its neighbours are untouched
  REQUIRE(data[4 * 3 + 1] == 104.0);
}

TEST_CASE("AttrVectView refuses a field list shorter than the vector",
          "[fields]") {
  // A field list shorter than the data buffer, as if a fixed-length buffer
  // had truncated a longer list.
  const auto truncated = FieldList::parse("Sa_z:Sa_u:Sa_v");
  std::vector<double> data(37 * 2);
  REQUIRE_THROWS_WITH(AttrVectView(data.data(), truncated, 37, 2),
                      Catch::Contains("37 real fields") &&
                          Catch::Contains("names 3") &&
                          Catch::Contains("truncated") &&
                          Catch::Contains("'Sa_v'"));
}

TEST_CASE("AttrVectView refuses a null buffer with points", "[fields]") {
  const auto layout = FieldList::parse("a");
  REQUIRE_THROWS_AS(AttrVectView(nullptr, layout, 1, 4),
                    std::invalid_argument);
  // A rank with no points is normal, and may well have no buffer.
  REQUIRE_NOTHROW(AttrVectView(nullptr, layout, 1, 0));
}

// ---------------------------------------------------------------------------
// CouplerBinding
// ---------------------------------------------------------------------------

TEST_CASE("An import binding pulls named fields and reports the rest",
          "[fields]") {
  const auto x2o = FieldList::parse("Foxx_taux:Foxx_tauy:Foxx_swnet:Sa_pslv");
  FieldSet fields(5);
  fields.add("Foxx_swnet");
  fields.add("Foxx_taux");
  fields.add("Foxx_rofl", -1.0);

  CouplerBinding bind(CouplerBinding::Direction::Import, fields,
                      {{"Foxx_swnet"}, {"Foxx_taux"},
                       {"Foxx_rofl", Need::Optional}},
                      x2o);

  REQUIRE(bind.bound() == std::vector<std::string>{"Foxx_swnet", "Foxx_taux"});
  REQUIRE(bind.absent() == std::vector<std::string>{"Foxx_rofl"});
  REQUIRE(bind.unbound() == std::vector<std::string>{"Foxx_tauy", "Sa_pslv"});

  auto data = fortran_attr_vect(4, 5);
  AttrVectView av(data.data(), x2o, 4, 5);
  bind.pull(av);

  // Foxx_swnet is row 2 of the coupler's list and the first field here.
  REQUIRE(fields.get("Foxx_swnet")[3] == 203.0);
  REQUIRE(fields.get("Foxx_taux")[4] == 4.0);
  // The optional field the coupler does not send kept its fill.
  REQUIRE(fields.get("Foxx_rofl")[0] == -1.0);

  const auto text = bind.summary();
  REQUIRE_THAT(text, Catch::Contains("2 of 4") &&
                         Catch::Contains("ignored: Foxx_tauy Sa_pslv"));
}

TEST_CASE("A required import the coupler does not send fails at bind time",
          "[fields]") {
  const auto x2o = FieldList::parse("So_t:Foxx_swnet");
  FieldSet fields(1);
  fields.add("Foxx_SWnet");
  fields.add("Foxx_lat");

  REQUIRE_THROWS_WITH(
      CouplerBinding(CouplerBinding::Direction::Import, fields,
                     {{"Foxx_SWnet"}, {"Foxx_lat"}}, x2o),
      Catch::Contains("does not send") &&
          Catch::Contains("Foxx_SWnet (did you mean Foxx_swnet?)") &&
          Catch::Contains("Foxx_lat") &&
          Catch::Contains("So_t:Foxx_swnet"));
}

TEST_CASE("An export binding pushes fields and zeroes what it lacks",
          "[fields]") {
  const auto o2x = FieldList::parse("So_t:So_s:So_u:Fioo_q");
  FieldSet fields(5);
  auto t = fields.add("So_t");
  auto s = fields.add("So_s");
  for (std::size_t i = 0; i < 5; ++i) {
    t[i] = 270.0 + static_cast<double>(i);
    s[i] = 35.0;
  }

  CouplerBinding bind(CouplerBinding::Direction::Export, fields,
                      {{"So_t"}, {"So_s"}}, o2x);
  REQUIRE(bind.unbound() == std::vector<std::string>{"So_u", "Fioo_q"});

  // A buffer full of whatever the last step left there.
  auto data = fortran_attr_vect(4, 5);
  AttrVectView av(data.data(), o2x, 4, 5);
  bind.push(av);

  REQUIRE(av.at(0, 3) == 273.0);
  REQUIRE(av.at(1, 4) == 35.0);
  for (std::size_t p = 0; p < 5; ++p) {
    REQUIRE(av.at(2, p) == 0.0);
    REQUIRE(av.at(3, p) == 0.0);
  }
  REQUIRE_THAT(bind.summary(), Catch::Contains("zeroed:  So_u Fioo_q"));
}

TEST_CASE("Exporting a field the coupler does not carry is a compset error",
          "[fields]") {
  const auto o2x = FieldList::parse("So_t:So_s");
  FieldSet fields(1);
  fields.add("So_t");
  fields.add("Si_ifrac");
  REQUIRE_THROWS_WITH(CouplerBinding(CouplerBinding::Direction::Export,
                                     fields, {{"So_t"}, {"Si_ifrac"}}, o2x),
                      Catch::Contains("wrong compset") &&
                          Catch::Contains("Si_ifrac"));

  // Unless the component says it can live without it.
  CouplerBinding optional(CouplerBinding::Direction::Export, fields,
                          {{"So_t"}, {"Si_ifrac", Need::Optional}}, o2x);
  REQUIRE(optional.absent() == std::vector<std::string>{"Si_ifrac"});
}

TEST_CASE("A binding refuses a spec for a field the component never made",
          "[fields]") {
  const auto list = FieldList::parse("So_t");
  FieldSet fields(1);
  REQUIRE_THROWS_AS(CouplerBinding(CouplerBinding::Direction::Import, fields,
                                   {{"So_t"}}, list),
                    std::logic_error);
}

TEST_CASE("A binding checks direction and point count every step",
          "[fields]") {
  const auto list = FieldList::parse("a:b");
  FieldSet fields(3);
  fields.add("a");
  CouplerBinding import(CouplerBinding::Direction::Import, fields, {{"a"}},
                        list);

  auto data = fortran_attr_vect(2, 3);
  AttrVectView av(data.data(), list, 2, 3);
  REQUIRE_THROWS_AS(import.push(av), std::logic_error);

  // The decomposition changed under the binding: 4 points, not 3.
  auto more = fortran_attr_vect(2, 4);
  AttrVectView av4(more.data(), list, 2, 4);
  REQUIRE_THROWS_WITH(import.pull(av4),
                      Catch::Contains("4 points") && Catch::Contains("have 3"));

  // A differently laid out vector with the same count.
  const auto swapped = FieldList::parse("b:a");
  AttrVectView av_swapped(data.data(), swapped, 2, 3);
  REQUIRE_THROWS_WITH(import.pull(av_swapped),
                      Catch::Contains("laid out differently"));
}

// ---------------------------------------------------------------------------
// MaskSet and masked exports
// ---------------------------------------------------------------------------

TEST_CASE("MaskSet holds binary masks and refuses anything else", "[fields]") {
  MaskSet masks(4);
  masks.set("ocean", std::vector<double>{1, 1, 1, 0});
  masks.set("sea_ice", std::vector<double>{1, 0, 0, 0});
  REQUIRE(masks.count("ocean") == 3);
  REQUIRE(masks.count("sea_ice") == 1);
  REQUIRE(masks.names() == std::vector<std::string>{"ocean", "sea_ice"});

  REQUIRE_THROWS_WITH(masks.set("ice_fraction",
                                std::vector<double>{0.9, 0, 0.2, 0}),
                      Catch::Contains("2 values") &&
                          Catch::Contains("point 0 is 0.9"));
  REQUIRE_THROWS_WITH(masks.set("short", std::vector<double>{1, 1}),
                      Catch::Contains("2 values for 4 points"));
  REQUIRE_THROWS_WITH(masks.get("land"), Catch::Contains("ocean, sea_ice"));
}

TEST_CASE("An export bounded by its channel's mask is zero outside it",
          "[fields]") {
  // Four ocean cells, sea ice valid on the first only: without masking, the
  // ice values held everywhere would reach the coupler as sea ice in the
  // tropics.
  const auto o2x = FieldList::parse("So_t:Si_ifrac");
  FieldSet fields(4);
  auto sst = fields.add("So_t");
  auto ice = fields.add("Si_ifrac");
  for (std::size_t i = 0; i < 4; ++i) {
    sst[i] = 300.0;
    ice[i] = 0.6;
  }
  MaskSet masks(4);
  masks.set("ocean", std::vector<double>{1, 1, 1, 1});
  masks.set("sea_ice", std::vector<double>{1, 0, 0, 0});

  CouplerBinding bind(CouplerBinding::Direction::Export, fields,
                      {{"So_t", Need::Required, "K", "ocean"},
                       {"Si_ifrac", Need::Required, "1", "sea_ice"}},
                      o2x, &masks);
  std::vector<double> av(2 * 4, -1.0);
  AttrVectView view(av.data(), o2x, 2, 4);
  bind.push(view);

  REQUIRE(view.at(1, 0) == 0.6);
  REQUIRE(view.at(1, 1) == 0.0);
  REQUIRE(view.at(1, 3) == 0.0);
  REQUIRE(view.at(0, 3) == 300.0);
  // The component's own array is untouched.
  REQUIRE(ice[3] == 0.6);
  REQUIRE_THAT(bind.summary(),
               Catch::Contains("masked:  So_t (ocean) Si_ifrac (sea_ice)"));
}

TEST_CASE("A mask that is named but not set fails at bind time", "[fields]") {
  const auto o2x = FieldList::parse("Si_ifrac");
  FieldSet fields(2);
  fields.add("Si_ifrac");
  MaskSet masks(2);

  REQUIRE_THROWS_WITH(
      CouplerBinding(CouplerBinding::Direction::Export, fields,
                     {{"Si_ifrac", Need::Required, "1", "sea_ice"}}, o2x,
                     &masks),
      Catch::Contains("mask 'sea_ice'") && Catch::Contains("not been set"));
  REQUIRE_THROWS_AS(
      CouplerBinding(CouplerBinding::Direction::Export, fields,
                     {{"Si_ifrac", Need::Required, "1", "sea_ice"}}, o2x),
      std::invalid_argument);
  REQUIRE_THROWS_WITH(
      CouplerBinding(CouplerBinding::Direction::Import, fields,
                     {{"Si_ifrac", Need::Required, "1", "sea_ice"}}, o2x,
                     &masks),
      Catch::Contains("only exports are masked"));
}

} // namespace test
} // namespace fields
} // namespace emulator
