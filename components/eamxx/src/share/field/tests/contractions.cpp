#include <catch2/catch.hpp>

#include "share/field/field.hpp"
#include "share/field/field_utils.hpp"
#include "share/util/eamxx_universal_constants.hpp"

#include "share/core/eamxx_setup_random_test.hpp"

#include <algorithm>
#include <limits>

namespace scream {

TEST_CASE("field_contractions") {
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  // Accumulations in the Kokkos threaded reductions may be done in a
  // different order than the manual ones below, so we can only test
  // correctness up to a tolerance
  auto tol = std::numeric_limits<Real>::epsilon() * 100;

  auto seed = get_random_test_seed();

  SECTION("horiz_contraction") {

    int ncol = 129;
    int ncmp = 4;
    int nlev = 17;

    // Set a weight field with values [1,2,...,ncols]
    FieldIdentifier wfid("f", {{COL}, {ncol}}, none, "g");
    Field w(wfid,true);
    randomize_uniform(w,seed++);

    // Create (random) sample fields
    // NOTE: suffix _x stands for "horizontal", _z for "vertical", _xz for "horiz-and-vert"
    FieldIdentifier scl_fid("s", {{}, {}}, none, "g");  // scalar
    FieldIdentifier vec_x_fid("v_x", {{COL, CMP}, {ncol, ncmp}}, none, "g");
    FieldIdentifier scl_xz_fid("s_xz", {{COL, LEV}, {ncol, nlev}}, none, "g");
    FieldIdentifier vec_xz_fid("v_xz", {{COL, CMP, LEV}, {ncol, ncmp, nlev}}, none, "g");
    Field scl(scl_fid,true);
    Field vec_x(vec_x_fid,true);
    Field vec_xz(vec_xz_fid,true);
    randomize_uniform(scl, seed++);
    randomize_uniform(vec_x, seed++);
    randomize_uniform(vec_xz, seed++);

    FieldIdentifier scl_x_fid("s_x", {{COL}, {ncol}}, none, "g");
    FieldIdentifier scl_z_fid("s_z", {{LEV}, {nlev}}, none, "g");
    FieldIdentifier vec_fid  ("v", {{CMP}, {ncmp}}, none, "g");
    FieldIdentifier vec_z_fid("v_z", {{CMP, LEV}, {ncmp, nlev}}, none, "g");

    Field scl_x(scl_x_fid);
    Field vec  (vec_fid);
    Field vec_z(vec_z_fid);

    // Test invalid inputs
    REQUIRE_THROWS(horiz_contraction(scl, scl_x, scl_x.clone()));  // input not allocated yet
    REQUIRE_THROWS(horiz_contraction(vec, vec_x, scl_x.clone()));  // output not allocated yet
    REQUIRE_THROWS(horiz_contraction(scl, scl_x.clone(), scl_x));  // weight not allocated yet

    scl_x.allocate_view();
    vec.allocate_view();
    vec_z.allocate_view();

    REQUIRE_THROWS(horiz_contraction(scl, vec_x, scl_x)); // incompatible input-output layout
    REQUIRE_THROWS(horiz_contraction(scl, scl_x, vec_x)); // incompatible weight layout

    SECTION ("scl_x") {
      auto f_in = scl_x.clone(CloneFlags::CopyData);
      auto f_out = scl.clone();

      horiz_contraction(f_out, f_in, w);

      f_out.sync_to_host();
      auto v_in  = f_in.get_view<const Real *, Host>();
      auto v_out = f_out.get_view<Real, Host>();
      auto v_w = w.get_view<const Real*, Host>();
      Real manual = 0;
      for(int i = 0; i < ncol; ++i) {
        manual += v_w(i) * v_in(i);
      }
      REQUIRE_THAT(v_out(), Catch::Matchers::WithinRel(manual, tol));
    }

    SECTION ("scl_x_masked") {
      auto f_in = scl_x.clone(CloneFlags::CopyData);
      auto f_out = scl.clone();
      auto mask = f_in.create_valid_mask("mask",Field::MaskInit::Valid);
      mask.sync_to_host();
      auto v_mask = mask.get_view<int *, Host>();
      v_mask(ncol - 1) = 0;
      mask.sync_to_dev();
      
      horiz_contraction(f_out, f_in, w);
      f_out.sync_to_host();
      auto v_in  = f_in.get_view<const Real *, Host>();
      auto v_out = f_out.get_view<Real, Host>();
      auto v_w = w.get_view<const Real*, Host>();
      Real manual = 0;
      for(int i = 0; i < ncol; ++i) {
        if (v_mask(i))
          manual += v_w(i) * v_in(i);
      }
      REQUIRE_THAT(v_out(), Catch::Matchers::WithinRel(manual, tol));

      // Repeat but with ALL masked values (all mask=0 => sum = 0)
      mask.deep_copy(0);
      horiz_contraction(f_out, f_in, w);
      f_out.sync_to_host();
      REQUIRE_THAT(v_out(), Catch::Matchers::WithinAbs(Real(0), tol));
    }

    SECTION ("vec_xz") {
      auto f_in  = vec_xz.clone(CloneFlags::CopyData);
      auto f_out = vec_z.clone();
      auto mask = f_in.create_valid_mask();
      randomize_uniform(mask,seed++);

      horiz_contraction(f_out,f_in,w);

      f_out.sync_to_host();
      auto v_in = f_in.get_view<Real ***, Host>();
      auto v_out = f_out.get_view<Real **, Host>();
      auto v_mask = mask.get_view<const int ***, Host>();
      auto v_w = w.get_view<const Real*, Host>();

      for(int k = 0; k < nlev; ++k) {
        for(int j = 0; j < ncmp; ++j) {
          Real manual = 0;
          for(int i = 0; i < ncol; ++i) {
            if (v_mask(i,j,k))
              manual += v_w(i) * v_in(i, j, k);
          }
          REQUIRE_THAT(v_out(j, k), Catch::Matchers::WithinRel(manual, tol));
        }
      }
    }
  }

  SECTION("vert_contraction") {
    int ncol = 129;
    int ncmp = 4;
    int nlev = 17;

    // Test inputs with both LEV and ILEV vert dim tags
    for(FieldTag lev_tag : {LEV,ILEV}) {
      // Create (random) sample fields
      // NOTE: suffix _x stands for "horizontal", _z for "vertical", _xz for "horiz-and-vert"
      FieldIdentifier scl_fid   ("s",    {{}, {}}, ekat::units::none, "g");  // scalar
      FieldIdentifier scl_xz_fid("s_xz", {{COL, lev_tag}, {ncol, nlev}}, ekat::units::none, "g");
      FieldIdentifier vec_z_fid ("v_z",  {{CMP, lev_tag}, {ncmp, nlev}}, ekat::units::none, "g");
      FieldIdentifier vec_xz_fid("v_xz", {{COL, CMP, lev_tag}, {ncol, ncmp, nlev}}, ekat::units::none, "g");

      Field scl(scl_fid,true);
      Field scl_xz(scl_xz_fid,true);
      Field vec_z (vec_z_fid,true);
      Field vec_xz(vec_xz_fid,true);
      randomize_uniform(scl, seed++);
      randomize_uniform(scl_xz, seed++);
      randomize_uniform(vec_z, seed++);
      randomize_uniform(vec_xz, seed++);

      // Output/utility fields
      FieldIdentifier scl_x_fid("s_x", {{COL}, {ncol}}, ekat::units::none, "g");
      FieldIdentifier vec_x_fid("v_x", {{COL,CMP}, {ncol,ncmp}}, ekat::units::none, "g");
      FieldIdentifier vec_fid  ("v", {{CMP}, {ncmp}}, ekat::units::none, "g");
      FieldIdentifier scl_z_fid ("s_z",  {{lev_tag}, {nlev}}, ekat::units::none, "g");

      Field scl_x(scl_x_fid,true);
      Field vec_x(vec_x_fid,true);
      Field scl_z(scl_z_fid,true);
      Field vec(vec_fid,true);

      // Test invalid inputs
      REQUIRE_THROWS(vert_contraction(scl_x, Field(vec_z_fid), scl_z));  // input not allocated yet
      REQUIRE_THROWS(vert_contraction(Field(scl_x_fid), vec_z, scl_z));  // output not allocated yet
      REQUIRE_THROWS(vert_contraction(scl_x, vec_z, Field(scl_z_fid)));  // weight not allocated yet

      REQUIRE_THROWS(vert_contraction(scl, vec_z, scl_z));  // incompatible input-output layout
      REQUIRE_THROWS(vert_contraction(scl, scl_z, vec_xz)); // incompatible weight layout

      SECTION ("scl_z") {
        auto f_in = scl_z.clone(CloneFlags::CopyData);
        auto f_out = scl.clone();
        auto w = scl_z.clone();
        randomize_uniform (w,seed++);

        vert_contraction(f_out,f_in,w);
        f_out.sync_to_host();

        auto v_in  = f_in.get_view<const Real*,Host>(); 
        auto v_out = f_out.get_view<const Real,Host>(); 
        auto v_w   = w.get_view<const Real*,Host>(); 

        Real manual = 0;
        for (int lev=0; lev<nlev; ++lev) {
          manual += v_w(lev)*v_in(lev);
        }
        REQUIRE_THAT(v_out(), Catch::Matchers::WithinRel(manual, tol));
      }

      SECTION ("scl_z_masked") {
        auto f_in = scl_z.clone(CloneFlags::CopyData);
        auto f_out = scl.clone();
        auto w = scl_z.clone();
        randomize_uniform (w,seed++);
        auto mask = f_in.create_valid_mask();
        randomize_uniform (mask,seed++);

        vert_contraction(f_out,f_in,w);
        f_out.sync_to_host();

        auto v_in   = f_in.get_view<const Real*,Host>(); 
        auto v_out  = f_out.get_view<const Real,Host>(); 
        auto v_w    = w.get_view<const Real*,Host>(); 
        auto v_mask = mask.get_view<const int*,Host>(); 

        Real manual = 0;
        for (int lev=0; lev<nlev; ++lev) {
          if (v_mask(lev)) {
            manual += v_w(lev)*v_in(lev);
          }
        }
        REQUIRE_THAT(v_out(), Catch::Matchers::WithinRel(manual, tol));

        mask.deep_copy(0);
        vert_contraction(f_out,f_in,w);
        f_out.sync_to_host();
        REQUIRE_THAT(v_out(), Catch::Matchers::WithinRel(0, tol));
      }

      SECTION ("scl_xz_w_z") {
        auto f_in = scl_xz.clone(CloneFlags::CopyData);
        auto f_out = scl_x.clone();
        auto w = scl_z.clone();
        randomize_uniform (w,seed++);
        auto mask = f_in.create_valid_mask();
        randomize_uniform (mask,seed++);

        vert_contraction(f_out,f_in,w);
        f_out.sync_to_host();

        auto v_in   = f_in.get_view<const Real**,Host>(); 
        auto v_out  = f_out.get_view<const Real*,Host>(); 
        auto v_w    = w.get_view<const Real*,Host>(); 
        auto v_mask = mask.get_view<const int**,Host>(); 

        for (int col=0; col<ncol; ++col) {
          Real manual = 0;
          for (int lev=0; lev<nlev; ++lev) {
            if (v_mask(col,lev)) {
              manual += v_w(lev)*v_in(col,lev);
            }
          }
          REQUIRE_THAT(v_out(col), Catch::Matchers::WithinRel(manual, tol));
        }
      }

      SECTION ("vec_xz_w_xz") {
        auto f_in = vec_xz.clone(CloneFlags::CopyData);
        auto f_out = vec_x.clone();
        auto w = scl_xz.clone();
        randomize_uniform (w,seed++);
        auto mask = f_in.create_valid_mask();
        randomize_uniform (mask,seed++);

        vert_contraction(f_out,f_in,w);
        f_out.sync_to_host();

        auto v_in   = f_in.get_view<const Real***,Host>(); 
        auto v_out  = f_out.get_view<const Real**,Host>(); 
        auto v_w    = w.get_view<const Real**,Host>(); 
        auto v_mask = mask.get_view<const int***,Host>(); 

        for (int col=0; col<ncol; ++col) {
          for (int cmp=0; cmp<ncmp; ++cmp) {
            Real manual = 0;
            for (int lev=0; lev<nlev; ++lev) {
              if (v_mask(col,cmp,lev)) {
                manual += v_w(col,lev)*v_in(col,cmp,lev);
              }
            }
            REQUIRE_THAT(v_out(col,cmp), Catch::Matchers::WithinRel(manual, tol));
          }
        }
      }
    }
  }
}

TEST_CASE("field_extrema") {
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  // Max/min involve no arithmetic, so we can compare exactly
  auto seed = get_random_test_seed();

  const int ncol = 129;
  const int ncmp = 4;
  const int nlev = 17;

  constexpr auto lowest  = std::numeric_limits<Real>::lowest();
  constexpr auto highest = std::numeric_limits<Real>::max();

  FieldIdentifier scl_fid   ("s",    {{}, {}}, none, "g");
  FieldIdentifier scl_x_fid ("s_x",  {{COL}, {ncol}}, none, "g");
  FieldIdentifier scl_z_fid ("s_z",  {{LEV}, {nlev}}, none, "g");
  FieldIdentifier vec_fid   ("v",    {{CMP}, {ncmp}}, none, "g");
  FieldIdentifier vec_x_fid ("v_x",  {{COL,CMP}, {ncol,ncmp}}, none, "g");
  FieldIdentifier vec_z_fid ("v_z",  {{CMP,LEV}, {ncmp,nlev}}, none, "g");
  FieldIdentifier scl_xz_fid("s_xz", {{COL,LEV}, {ncol,nlev}}, none, "g");
  FieldIdentifier vec_xz_fid("v_xz", {{COL,CMP,LEV}, {ncol,ncmp,nlev}}, none, "g");

  SECTION ("invalid_inputs") {
    Field scl(scl_fid,true);
    Field scl_x(scl_x_fid,true);
    Field scl_z(scl_z_fid,true);
    Field vec_x(vec_x_fid,true);

    REQUIRE_THROWS(vert_max(scl, Field(scl_z_fid)));   // input not allocated yet
    REQUIRE_THROWS(horiz_min(Field(scl_fid), scl_x));  // output not allocated yet
    REQUIRE_THROWS(vert_max(scl, vec_x));              // input does not end with LEV
    REQUIRE_THROWS(horiz_min(scl, vec_x));             // incompatible input-output layout
    REQUIRE_THROWS(horiz_max(scl, scl_z));             // input does not start with COL
  }

  SECTION ("vert") {
    // rank-1 input, masked: includes the all-masked case
    Field s_z(scl_z_fid,true);
    Field s(scl_fid,true);
    randomize_uniform(s_z,seed++,-1,1);
    auto& m_z = s_z.create_valid_mask();
    randomize_uniform(m_z,seed++);
    m_z.sync_to_host();
    s_z.sync_to_host();
    {
      auto v_in = s_z.get_view<const Real*,Host>();
      auto v_m  = m_z.get_view<const int*,Host>();
      Real mx = lowest, mn = highest;
      for (int k=0; k<nlev; ++k) {
        if (v_m(k)) {
          mx = std::max(mx,v_in(k));
          mn = std::min(mn,v_in(k));
        }
      }
      vert_max(s,s_z);
      s.sync_to_host();
      REQUIRE (s.get_view<const Real,Host>()()==mx);
      vert_min(s,s_z);
      s.sync_to_host();
      REQUIRE (s.get_view<const Real,Host>()()==mn);

      m_z.deep_copy(0);
      vert_max(s,s_z);
      s.sync_to_host();
      REQUIRE (s.get_view<const Real,Host>()()==lowest);
      vert_min(s,s_z);
      s.sync_to_host();
      REQUIRE (s.get_view<const Real,Host>()()==highest);
    }

    // rank-2 input, unmasked
    Field s_xz(scl_xz_fid,true);
    Field s_x(scl_x_fid,true);
    randomize_uniform(s_xz,seed++,-1,1);
    s_xz.sync_to_host();
    {
      auto v_in = s_xz.get_view<const Real**,Host>();
      auto v_out = s_x.get_view<const Real*,Host>();
      vert_max(s_x,s_xz);
      s_x.sync_to_host();
      for (int i=0; i<ncol; ++i) {
        Real mx = lowest;
        for (int k=0; k<nlev; ++k) {
          mx = std::max(mx,v_in(i,k));
        }
        REQUIRE (v_out(i)==mx);
      }
      vert_min(s_x,s_xz);
      s_x.sync_to_host();
      for (int i=0; i<ncol; ++i) {
        Real mn = highest;
        for (int k=0; k<nlev; ++k) {
          mn = std::min(mn,v_in(i,k));
        }
        REQUIRE (v_out(i)==mn);
      }
    }

    // rank-3 input, masked
    Field v_xz(vec_xz_fid,true);
    Field v_x(vec_x_fid,true);
    randomize_uniform(v_xz,seed++,-1,1);
    auto& m_xz = v_xz.create_valid_mask();
    randomize_uniform(m_xz,seed++);
    v_xz.sync_to_host();
    m_xz.sync_to_host();
    {
      auto v_in  = v_xz.get_view<const Real***,Host>();
      auto v_m   = m_xz.get_view<const int***,Host>();
      auto v_out = v_x.get_view<const Real**,Host>();
      vert_max(v_x,v_xz);
      v_x.sync_to_host();
      for (int i=0; i<ncol; ++i) {
        for (int j=0; j<ncmp; ++j) {
          Real mx = lowest;
          for (int k=0; k<nlev; ++k) {
            if (v_m(i,j,k)) {
              mx = std::max(mx,v_in(i,j,k));
            }
          }
          REQUIRE (v_out(i,j)==mx);
        }
      }
      vert_min(v_x,v_xz);
      v_x.sync_to_host();
      for (int i=0; i<ncol; ++i) {
        for (int j=0; j<ncmp; ++j) {
          Real mn = highest;
          for (int k=0; k<nlev; ++k) {
            if (v_m(i,j,k)) {
              mn = std::min(mn,v_in(i,j,k));
            }
          }
          REQUIRE (v_out(i,j)==mn);
        }
      }
    }
  }

  SECTION ("horiz") {
    // rank-1 input, unmasked
    Field s_x(scl_x_fid,true);
    Field s(scl_fid,true);
    randomize_uniform(s_x,seed++,-1,1);
    s_x.sync_to_host();
    {
      auto v_in = s_x.get_view<const Real*,Host>();
      Real mx = lowest, mn = highest;
      for (int i=0; i<ncol; ++i) {
        mx = std::max(mx,v_in(i));
        mn = std::min(mn,v_in(i));
      }
      horiz_max(s,s_x);
      s.sync_to_host();
      REQUIRE (s.get_view<const Real,Host>()()==mx);
      horiz_min(s,s_x);
      s.sync_to_host();
      REQUIRE (s.get_view<const Real,Host>()()==mn);
    }

    // rank-2 input, masked, with one fully masked entry
    Field v_x(vec_x_fid,true);
    Field v(vec_fid,true);
    randomize_uniform(v_x,seed++,-1,1);
    auto& m_x = v_x.create_valid_mask();
    randomize_uniform(m_x,seed++);
    m_x.sync_to_host();
    auto v_m = m_x.get_view<int**,Host>();
    for (int i=0; i<ncol; ++i) {
      v_m(i,0) = 0;
    }
    m_x.sync_to_dev();
    v_x.sync_to_host();
    {
      auto v_in  = v_x.get_view<const Real**,Host>();
      auto v_out = v.get_view<const Real*,Host>();
      horiz_max(v,v_x);
      v.sync_to_host();
      for (int j=0; j<ncmp; ++j) {
        Real mx = lowest;
        for (int i=0; i<ncol; ++i) {
          if (v_m(i,j)) {
            mx = std::max(mx,v_in(i,j));
          }
        }
        REQUIRE (v_out(j)==mx);
      }
      REQUIRE (v_out(0)==lowest);
      horiz_min(v,v_x);
      v.sync_to_host();
      for (int j=0; j<ncmp; ++j) {
        Real mn = highest;
        for (int i=0; i<ncol; ++i) {
          if (v_m(i,j)) {
            mn = std::min(mn,v_in(i,j));
          }
        }
        REQUIRE (v_out(j)==mn);
      }
      REQUIRE (v_out(0)==highest);
    }

    // rank-3 input, unmasked
    Field v_xz(vec_xz_fid,true);
    Field v_z(vec_z_fid,true);
    randomize_uniform(v_xz,seed++,-1,1);
    v_xz.sync_to_host();
    {
      auto v_in  = v_xz.get_view<const Real***,Host>();
      auto v_out = v_z.get_view<const Real**,Host>();
      horiz_max(v_z,v_xz);
      v_z.sync_to_host();
      for (int j=0; j<ncmp; ++j) {
        for (int k=0; k<nlev; ++k) {
          Real mx = lowest;
          for (int i=0; i<ncol; ++i) {
            mx = std::max(mx,v_in(i,j,k));
          }
          REQUIRE (v_out(j,k)==mx);
        }
      }
      horiz_min(v_z,v_xz);
      v_z.sync_to_host();
      for (int j=0; j<ncmp; ++j) {
        for (int k=0; k<nlev; ++k) {
          Real mn = highest;
          for (int i=0; i<ncol; ++i) {
            mn = std::min(mn,v_in(i,j,k));
          }
          REQUIRE (v_out(j,k)==mn);
        }
      }
    }
  }
}

} // namespace scream
