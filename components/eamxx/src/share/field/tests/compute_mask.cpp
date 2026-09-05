#include <catch2/catch.hpp>

#include "share/field/field.hpp"
#include "share/field/field_utils.hpp"

namespace scream {

TEST_CASE ("compute_mask") {
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  const int ncols = 10;
  const int nlevs = 128;

  // Create fields
  std::vector<FieldTag> tags3d = {COL, CMP, LEV};
  std::vector<FieldTag> tags2d = {COL, LEV};
  std::vector<int>      dims3d = {ncols,2,nlevs};
  std::vector<int>      dims2d = {ncols,nlevs};

  FieldIdentifier fid0d ("foo", {}, none, "some_grid");
  FieldIdentifier fid0di ("foo", {}, none, "some_grid",DataType::IntType);
  FieldIdentifier fid3d ("foo", {tags3d,dims3d}, none, "some_grid");
  FieldIdentifier fid3di ("foo", {tags3d,dims3d}, none, "some_grid", DataType::IntType);
  FieldIdentifier fid2d ("foo", {tags2d,dims2d}, none, "some_grid");

  SECTION ("exceptions") {
    // Test compute_mask exception handling
    Field f (fid3d);
    Field m1 (fid3d);

    REQUIRE_THROWS(compute_mask(f,1,Comparison::EQ,m1)); // Field not allocated
    f.allocate_view();
    REQUIRE_THROWS(compute_mask(f,1,Comparison::EQ,m1)); // Mask not allocated
    m1.allocate_view();

    Field m2 (fid2d);
    m2.allocate_view();
    REQUIRE_THROWS(compute_mask(f,1,Comparison::EQ,m2)); // incompatible layouts

    // Comparing two fields with DIFFERENT units is meaningless, and both
    // operands carry units, so it is rejected.
    Field mask_i (fid3di);
    mask_i.allocate_view();
    FieldIdentifier fid3d_K ("bar", {tags3d,dims3d}, K, "some_grid");
    FieldIdentifier fid3d_Pa("baz", {tags3d,dims3d}, Pa, "some_grid");
    Field f_K (fid3d_K), f_Pa (fid3d_Pa);
    f_K.allocate_view();
    f_Pa.allocate_view();
    REQUIRE_THROWS(compute_mask(f_K,f_Pa,Comparison::GT,mask_i)); // K vs Pa

    // Same units compare fine...
    FieldIdentifier fid3d_K2 ("qux", {tags3d,dims3d}, K, "some_grid");
    Field f_K2 (fid3d_K2);
    f_K2.allocate_view();
    REQUIRE_NOTHROW(compute_mask(f_K,f_K2,Comparison::GT,mask_i));

    // ...and a field that never declared units is not evidence of a
    // mismatch, so it is compared without complaint.
    FieldIdentifier fid3d_inv ("quux", {tags3d,dims3d}, Units::invalid(), "some_grid");
    Field f_inv (fid3d_inv);
    f_inv.allocate_view();
    REQUIRE_NOTHROW(compute_mask(f_K,f_inv,Comparison::GT,mask_i));

    // A scalar threshold carries no units at all: it is taken in the field's
    // own units and cannot be checked. It must still work.
    REQUIRE_NOTHROW(compute_mask(f_K,273.15,Comparison::LT,mask_i));
  }

  SECTION ("check") {
    Field x(fid3d), one(fid3di), zero(fid3di), m(fid3di);

    x.allocate_view();
    one.allocate_view();
    m.allocate_view();
    zero.allocate_view();

    one.deep_copy(1);
    zero.deep_copy(0);
    x.deep_copy(2);

    // x==1 is false
    m.deep_copy(-1);
    compute_mask(x,1,Comparison::EQ,m);
    REQUIRE(views_are_equal(m,zero));

    // x!=1 is true
    m.deep_copy(-1);
    compute_mask(x,1,Comparison::NE,m);
    REQUIRE(views_are_equal(m,one));

    // x==2 is true
    m.deep_copy(-1);
    compute_mask(x,2,Comparison::EQ,m);
    REQUIRE(views_are_equal(m,one));

    // x>1 is true
    m.deep_copy(-1);
    compute_mask(x,1,Comparison::GT,m);
    REQUIRE(views_are_equal(m,one));

    // x>2 is false
    m.deep_copy(-1);
    compute_mask(x,2,Comparison::GT,m);
    REQUIRE(views_are_equal(m,zero));

    // x>=2 is true
    m.deep_copy(-1);
    compute_mask(x,2,Comparison::GE,m);
    REQUIRE(views_are_equal(m,one));

    // x<3 is true
    m.deep_copy(-1);
    compute_mask(x,3,Comparison::LT,m);
    REQUIRE(views_are_equal(m,one));

    // x<2 is false
    m.deep_copy(-1);
    compute_mask(x,2,Comparison::LT,m);
    REQUIRE(views_are_equal(m,zero));

    // x<=2 is true
    m.deep_copy(-1);
    compute_mask(x,2,Comparison::LE,m);
    REQUIRE(views_are_equal(m,one));
  }
  SECTION ("check 0d") {
    Field x(fid0d), one(fid0di), zero(fid0di), m(fid0di);

    x.allocate_view();
    one.allocate_view();
    m.allocate_view();
    zero.allocate_view();

    one.deep_copy(1);
    zero.deep_copy(0);
    x.deep_copy(2);

    // x==1 is false
    m.deep_copy(-1);
    compute_mask(x,1,Comparison::EQ,m);
    REQUIRE(views_are_equal(m,zero));

    // x!=1 is true
    m.deep_copy(-1);
    compute_mask(x,1,Comparison::NE,m);
    REQUIRE(views_are_equal(m,one));

    // x==2 is true
    m.deep_copy(-1);
    compute_mask(x,2,Comparison::EQ,m);
    REQUIRE(views_are_equal(m,one));

    // x>1 is true
    m.deep_copy(-1);
    compute_mask(x,1,Comparison::GT,m);
    REQUIRE(views_are_equal(m,one));

    // x>2 is false
    m.deep_copy(-1);
    compute_mask(x,2,Comparison::GT,m);
    REQUIRE(views_are_equal(m,zero));

    // x>=2 is true
    m.deep_copy(-1);
    compute_mask(x,2,Comparison::GE,m);
    REQUIRE(views_are_equal(m,one));

    // x<3 is true
    m.deep_copy(-1);
    compute_mask(x,3,Comparison::LT,m);
    REQUIRE(views_are_equal(m,one));

    // x<2 is flase
    m.deep_copy(-1);
    compute_mask(x,2,Comparison::LT,m);
    REQUIRE(views_are_equal(m,zero));

    // x<=2 is true
    m.deep_copy(-1);
    compute_mask(x,2,Comparison::LE,m);
    REQUIRE(views_are_equal(m,one));
  }
}

} // namespace scream
