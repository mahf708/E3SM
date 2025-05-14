#include <catch2/catch.hpp>

#include "share/tests/diags_redux_tests/diags_redux_procs_tests.hpp"

namespace scream {

TEST_CASE("diags_redux_constructor", "[diags_redux]") {
  const ekat::Comm comm;
  ekat::ParameterList params;

  std::string param1_val = "value1";
  params.set("param1", param1_val);

  auto diags_redux = std::make_shared<TestDiagsRedux>(comm, params);

  // Verify the constructor
  REQUIRE(diags_redux != nullptr);

  REQUIRE(diags_redux->get_comm().size() == comm.size());
  REQUIRE(diags_redux->get_comm().rank() == comm.rank());

  // Correct usage of get with explicit type
  REQUIRE(diags_redux->get_params().get<std::string>("param1") == param1_val);
}

TEST_CASE("diags_redux_add_field") {
  ekat::Comm comm;
  ekat::ParameterList params;
  auto diags_redux = std::make_shared<TestDiagsRedux>(comm, params);
  using namespace ShortFieldTagsNames;
  FieldLayout layout({{COL, LEV}, {10, 20}});
  const auto units       = ekat::units::K;
  std::string grid_name  = "grid1";
  std::string field_name = "T_mid";
  diags_redux->add_field<Required>(field_name, layout, units, grid_name);
  // Verify that the field was added as required correctly
  auto req = diags_redux->get_required_field_requests().front();
  REQUIRE(req.fid.name() == field_name);
  REQUIRE(req.fid.get_units() == units);
  REQUIRE(req.fid.get_layout() == layout);
  REQUIRE(req.fid.get_grid_name() == grid_name);

  // Add a computed field
  std::string field_name2 = "T_int";
  diags_redux->add_field<Computed>(field_name2, layout, units, grid_name);
  // Verify that the field was added as computed correctly
  auto comp = diags_redux->get_computed_field_requests().front();
  REQUIRE(comp.fid.name() == field_name2);
  REQUIRE(comp.fid.get_units() == units);
  REQUIRE(comp.fid.get_layout() == layout);
  REQUIRE(comp.fid.get_grid_name() == grid_name);
}

TEST_CASE("diags_redux_set_fields") {
  ekat::Comm comm;
  ekat::ParameterList params;
  auto diags_redux = std::make_shared<TestDiagsRedux>(comm, params);
  using namespace ShortFieldTagsNames;
  FieldLayout layout({{COL, LEV}, {10, 20}});
  const auto units       = ekat::units::K;
  std::string grid_name  = "grid";
  std::string field_name = "T_mid";
  diags_redux->add_field<Required>(field_name, layout, units, grid_name);
  FieldIdentifier fid(field_name, layout, units, grid_name);
  Field field(fid);
  diags_redux->set_required_field(field);
  // Verify that the field was set correctly
  REQUIRE(diags_redux->get_fields_in().front().get_header().get_identifier().name() == field_name);
  REQUIRE(diags_redux->get_fields_in().front().get_header().get_identifier().get_units() == units);
  REQUIRE(diags_redux->get_fields_in().front().get_header().get_identifier().get_layout() == layout);
  REQUIRE(diags_redux->get_fields_in().front().get_header().get_identifier().get_grid_name() == grid_name);
}

} // namespace scream
