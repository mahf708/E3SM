#include <catch2/catch.hpp>

#include "share/remap/horizontal_remapper.hpp"
#include "share/grid/point_grid.hpp"
#include "share/scorpio_interface/eamxx_scorpio_interface.hpp"
#include "share/core/eamxx_setup_random_test.hpp"
#include "share/util/eamxx_utils.hpp"
#include "share/field/field_utils.hpp"
#include "share/util/eamxx_universal_constants.hpp"

#include <vector>
#include <utility>
#include <string>

namespace scream {

Field create_field (const std::string& name, const LayoutType lt, const AbstractGrid& grid)
{
  using namespace ShortFieldTagsNames;
  const auto& gn = grid.name();
  const auto  ndims = 2;
  Field f;
  switch (lt) {
    case LayoutType::Scalar1D:
      f = Field(FieldIdentifier(name,grid.get_vertical_layout(LEV),ekat::units::none,gn));  break;
    case LayoutType::Scalar2D:
      f = Field(FieldIdentifier(name,grid.get_2d_scalar_layout(),ekat::units::none,gn));  break;
    case LayoutType::Vector2D:
      f = Field(FieldIdentifier(name,grid.get_2d_vector_layout(ndims),ekat::units::none,gn));  break;
    case LayoutType::Scalar3D:
      f = Field(FieldIdentifier(name,grid.get_3d_scalar_layout(LEV),ekat::units::none,gn));
      f.get_header().get_alloc_properties().request_allocation(SCREAM_PACK_SIZE);  break;
    case LayoutType::Vector3D:
      f = Field(FieldIdentifier(name,grid.get_3d_vector_layout(ILEV,ndims),ekat::units::none,gn));
      f.get_header().get_alloc_properties().request_allocation(SCREAM_PACK_SIZE);  break;
    case LayoutType::Tensor3D:
      // rank 4: (COL,CMP,CMP,LEV). Needed to exercise the 'case 4' branch of the matvec kernels
      f = Field(FieldIdentifier(name,grid.get_3d_tensor_layout(LEV,{ndims,ndims}),ekat::units::none,gn));
      f.get_header().get_alloc_properties().request_allocation(SCREAM_PACK_SIZE);  break;
    default:
      EKAT_ERROR_MSG ("Invalid layout type for this unit test.\n");
  }
  f.allocate_view();

  return f;
}

Field create_field (const std::string& name, const LayoutType lt, const AbstractGrid& grid, int seed) {
  auto f = create_field(name,lt,grid);

  // Use discrete_distribution to get an integer, then use that as exponent for 2^-n.
  // This guarantees numbers that are exactly represented as FP numbers, which ensures
  // the test will produce the expected answer, regardless of how math ops are performed.
  std::vector<Real> values = {1,2,4,8,16,32,64,128,256,512};
  randomize_discrete(f,seed,values);
  return f;
}

Field all_gather_field (const Field& f, const ekat::Comm& comm) {
  constexpr auto COL = ShortFieldTagsNames::COL;
  const auto& fid = f.get_header().get_identifier();
  const auto& fl  = fid.get_layout();
  if (not fl.has_tag(COL)) {
    // Not partitioned
    return f;
  }
  int col_size = fl.clone().strip_dim(COL).size();
  auto tags = fl.tags();
  auto dims = fl.dims();
  int my_cols = dims[0];;
  comm.all_reduce(&my_cols, &dims.front(), 1, MPI_SUM );
  FieldLayout gfl(tags,dims);
  FieldIdentifier gfid("g" + f.name(),gfl,fid.get_units(),fid.get_grid_name(),fid.data_type());
  Field gf(gfid);
  gf.allocate_view();
  f.sync_to_host();
  std::vector<Real> data_vec(col_size);
  for (int pid=0,offset=0; pid<comm.size(); ++pid) {
    Real* data;
    int ncols = fl.dims()[0];
    comm.broadcast(&ncols,1,pid);
    for (int icol=0; icol<ncols; ++icol,offset+=col_size) {
      switch (fl.rank()) {
        case 1:
          if (pid==comm.rank()) {
            data = ekat::subview(f.get_view<Real*,Host>(),icol).data();
          } else {
            data = data_vec.data();
          }
          break;
        case 2:
          if (pid==comm.rank()) {
            data = ekat::subview(f.get_view<Real**,Host>(),icol).data();
          } else {
            data = data_vec.data();
          }
          break;
        case 3:
          if (pid==comm.rank()) {
            data = ekat::subview(f.get_view<Real***,Host>(),icol).data();
          } else {
            data = data_vec.data();
          }
          break;
        default:
          EKAT_ERROR_MSG (
              "Unexpected rank in RefiningRemapper unit test.\n"
              "  - field name: " + f.name() + "\n");
      }
      comm.broadcast(data,col_size,pid);
      auto gdata = gf.get_internal_view_data<Real,Host>()+offset;
      std::copy(data,data+col_size,gdata);
    }
  }
  gf.sync_to_dev();
  return gf;
}

void write_map_file (const std::string& filename, const int ngdofs_src) {
  // Add a dof in the middle of two coarse dofs
  const int ngdofs_tgt = 2*ngdofs_src-1;

  // Existing dofs are "copied", added dofs are averaged from neighbors
  const int nnz = ngdofs_src + 2*(ngdofs_src-1);

  scorpio::register_file(filename, scorpio::FileMode::Write);

  scorpio::define_dim(filename, "n_a", ngdofs_src);
  scorpio::define_dim(filename, "n_b", ngdofs_tgt);
  scorpio::define_dim(filename, "n_s", nnz);

  scorpio::define_var(filename, "col", {"n_s"}, "int");
  scorpio::define_var(filename, "row", {"n_s"}, "int");
  scorpio::define_var(filename, "S",   {"n_s"}, "double");

  scorpio::enddef(filename);

  std::vector<int> col(nnz), row(nnz);
  std::vector<double> S(nnz);
  int gid_base = 1;
  for (int i=0; i<ngdofs_src; ++i) {
    col[i] = i+gid_base;
    row[i] = i+gid_base;
      S[i] = 1.0;
  }
  for (int i=0; i<ngdofs_src-1; ++i) {
    col[ngdofs_src+2*i] = i+gid_base;
    row[ngdofs_src+2*i] = ngdofs_src+i+gid_base;
      S[ngdofs_src+2*i] = 0.5;

    col[ngdofs_src+2*i+1] = i+1+gid_base;
    row[ngdofs_src+2*i+1] = ngdofs_src+i+gid_base;
      S[ngdofs_src+2*i+1] = 0.5;
  }

  scorpio::write_var(filename,"row",row.data());
  scorpio::write_var(filename,"col",col.data());
  scorpio::write_var(filename,"S",  S.data());

  scorpio::release_file(filename);
}

// Write a map file that does NOT cover every tgt column: only the "copy" columns get
// weights, the interpolated ones are left unmapped. This is what you get from a
// regional tgt grid that is not fully contained in the src grid, or from a map built
// with unmapped destination cells dropped.
void write_partial_map_file (const std::string& filename, const int ngdofs_src) {
  const int ngdofs_tgt = 2*ngdofs_src-1;
  const int nnz = ngdofs_src;   // only the copies; no entries for the added dofs

  scorpio::register_file(filename, scorpio::FileMode::Write);

  scorpio::define_dim(filename, "n_a", ngdofs_src);
  scorpio::define_dim(filename, "n_b", ngdofs_tgt);
  scorpio::define_dim(filename, "n_s", nnz);

  scorpio::define_var(filename, "col", {"n_s"}, "int");
  scorpio::define_var(filename, "row", {"n_s"}, "int");
  scorpio::define_var(filename, "S",   {"n_s"}, "double");

  scorpio::enddef(filename);

  std::vector<int> col(nnz), row(nnz);
  std::vector<double> S(nnz);
  const int gid_base = 1;
  for (int i=0; i<ngdofs_src; ++i) {
    col[i] = i+gid_base;
    row[i] = i+gid_base;
      S[i] = 1.0;
  }

  scorpio::write_var(filename,"row",row.data());
  scorpio::write_var(filename,"col",col.data());
  scorpio::write_var(filename,"S",  S.data());

  scorpio::release_file(filename);
}

TEST_CASE ("refining_remapper") {
  using gid_type = AbstractGrid::gid_type;

  auto& catch_capture = Catch::getResultCapture();

  ekat::Comm comm(MPI_COMM_WORLD);

  int seed = get_random_test_seed(&comm);

  scorpio::init_subsystem(comm);

  // Create a map file
  const int ngdofs_src = 4*comm.size();
  const int ngdofs_tgt = 2*ngdofs_src-1;
  auto filename = "rr_tests_map.np" + std::to_string(comm.size()) + ".nc";
  write_map_file(filename,ngdofs_src);

  // Create source/target grid. For src grid, gids are fine. For tgt, create_point_grid
  // will partition gids, so that rank0 owns [0,n0), rank2 [n0,n0+n1),... But in the test
  // phase, we assume each rank owns a contiguous portion of the "1d segment", so we need
  // to modify the list of gids owned
  // Also, create tgt_grid 0-based to make modular arithmetic easier, then add 1 when fixing dofs
  const int nlevs = std::max(SCREAM_PACK_SIZE,16);
  auto src_grid = create_point_grid("src",ngdofs_src,nlevs,comm,1);
  auto tgt_grid = create_point_grid("tgt",ngdofs_tgt,nlevs,comm,0);
  auto dofs_h = tgt_grid->get_dofs_gids().get_view<gid_type*,Host>();
  for (int i=0; i<tgt_grid->get_num_local_dofs(); ++i) {
    int q = dofs_h[i] / 2;
    if (dofs_h[i] % 2 == 0) {
      dofs_h[i] = q + 1;
    } else {
      dofs_h[i] = ngdofs_src + q + 1;
    }
  }
  tgt_grid->get_dofs_gids().sync_to_dev();

  // Test bad usage, since they corrupt the remapper state for later
  {
    // Incompatible nlevs
    auto bad_src_grid1 = src_grid->clone("bad_src",true);
    bad_src_grid1->reset_vertical_configuration(nlevs+1,bad_src_grid1->get_vkind());
    CHECK_THROWS (std::make_shared<HorizontalRemapper>(bad_src_grid1,tgt_grid,filename));

    // src incompatible with map file
    auto bad_src_grid2 = create_point_grid("src",ngdofs_src+1,nlevs,comm);
    CHECK_THROWS (std::make_shared<HorizontalRemapper>(bad_src_grid2,tgt_grid,filename));

    // tgt incompatible with map file
    auto bad_tgt_grid1 = create_point_grid("tgt",ngdofs_tgt+1,nlevs,comm);
    CHECK_THROWS (std::make_shared<HorizontalRemapper>(src_grid,bad_tgt_grid1,filename));

    auto r = std::make_shared<HorizontalRemapper>(src_grid,tgt_grid,filename);
    auto src_grid = r->get_src_grid();
    Field bad_src(FieldIdentifier("",src_grid->get_2d_scalar_layout(),ekat::units::m,src_grid->name(),DataType::IntType));
    Field bad_tgt(FieldIdentifier("",tgt_grid->get_2d_scalar_layout(),ekat::units::m,tgt_grid->name(),DataType::IntType));
    // Fields don't need to be allocated prior to registration
    r->register_field(bad_src,bad_tgt);
    bad_src.allocate_view();
    bad_tgt.allocate_view();
    CHECK_THROWS (r->registration_ends()); // bad data type (must be real)
  }

  auto r = std::make_shared<HorizontalRemapper>(src_grid,tgt_grid,filename);

  auto bundle_src = create_field("bundle3d_src",LayoutType::Vector3D,*src_grid,seed);
  auto s1d_src   = create_field("s1d_src",LayoutType::Scalar1D,*src_grid,seed++);
  auto s2d_src   = create_field("s2d_src",LayoutType::Scalar2D,*src_grid,seed++);
  auto v2d_src   = create_field("v2d_src",LayoutType::Vector2D,*src_grid,seed++);
  auto s3d_src   = create_field("s3d_src",LayoutType::Scalar3D,*src_grid,seed++);
  auto v3d_src   = create_field("v3d_src",LayoutType::Vector3D,*src_grid,seed++);

  auto bundle_tgt = create_field("bundle3d_tgt",LayoutType::Vector3D,*tgt_grid);
  auto s1d_tgt   = create_field("s1d_tgt",LayoutType::Scalar1D,*tgt_grid);
  auto s2d_tgt   = create_field("s2d_tgt",LayoutType::Scalar2D,*tgt_grid);
  auto v2d_tgt   = create_field("v2d_tgt",LayoutType::Vector2D,*tgt_grid);
  auto s3d_tgt   = create_field("s3d_tgt",LayoutType::Scalar3D,*tgt_grid);
  auto v3d_tgt   = create_field("v3d_tgt",LayoutType::Vector3D,*tgt_grid);

  r->register_field(s1d_src,s1d_tgt);
  r->register_field(s2d_src,s2d_tgt);
  r->register_field(v2d_src,v2d_tgt);
  r->register_field(s3d_src,s3d_tgt);
  r->register_field(v3d_src,v3d_tgt);
  r->register_field(bundle_src.get_component(0),bundle_tgt.get_component(0));
  r->register_field(bundle_src.get_component(1),bundle_tgt.get_component(1));
  r->registration_ends();

  // Run remap
  CHECK_THROWS (r->remap_bwd()); // No backward remap
  r->remap_fwd();

  // Gather global copies (to make checks easier) and check src/tgt fields
  auto gs1d_src = all_gather_field(s1d_src,comm);
  auto gs2d_src = all_gather_field(s2d_src,comm);
  auto gv2d_src = all_gather_field(v2d_src,comm);
  auto gs3d_src = all_gather_field(s3d_src,comm);
  auto gv3d_src = all_gather_field(v3d_src,comm);
  auto gbundle_src = all_gather_field(bundle_src,comm);

  auto gs1d_tgt = all_gather_field(s1d_tgt,comm);
  auto gs2d_tgt = all_gather_field(s2d_tgt,comm);
  auto gv2d_tgt = all_gather_field(v2d_tgt,comm);
  auto gs3d_tgt = all_gather_field(s3d_tgt,comm);
  auto gv3d_tgt = all_gather_field(v3d_tgt,comm);
  auto gbundle_tgt = all_gather_field(bundle_tgt,comm);

  Real avg;
  // Scalar 1D
  {
    if (comm.am_i_root()) {
      printf(" -> Checking 1d scalars .........\n");
    }
    bool ok = true;
    gs1d_src.sync_to_host();
    gs1d_tgt.sync_to_host();

    CHECK (views_are_equal(gs1d_src,gs1d_tgt));
    ok &= catch_capture.lastAssertionPassed();
    if (comm.am_i_root()) {
      printf(" -> Checking 1d scalars ......... %s\n",ok ? "PASS" : "FAIL");
    }
  }

  // Scalar 2D
  {
    if (comm.am_i_root()) {
      printf(" -> Checking 2d scalars .........\n");
    }
    bool ok = true;
    gs2d_src.sync_to_host();
    gs2d_tgt.sync_to_host();

    auto src_v = gs2d_src.get_view<const Real*,Host>();
    auto tgt_v = gs2d_tgt.get_view<const Real*,Host>();

    // Coarse grid cols are just copied
    for (int icol=0; icol<ngdofs_src; ++icol) {
      CHECK (tgt_v[2*icol]==src_v[icol]);
      ok &= catch_capture.lastAssertionPassed();
    }
    // Fine cols are an average of the two cols nearby
    for (int icol=0; icol<ngdofs_src-1; ++icol) {
      avg = (src_v[icol] + src_v[icol+1]) / 2;
      CHECK (tgt_v[2*icol+1]==avg);
      ok &= catch_capture.lastAssertionPassed();
    }
    if (comm.am_i_root()) {
      printf(" -> Checking 2d scalars ......... %s\n",ok ? "PASS" : "FAIL");
    }
  }

  // Vector 2D
  {
    if (comm.am_i_root()) {
      printf(" -> Checking 2d vectors .........\n");
    }
    bool ok = true;
    gv2d_src.sync_to_host();
    gv2d_tgt.sync_to_host();

    auto src_v = gv2d_src.get_view<const Real**,Host>();
    auto tgt_v = gv2d_tgt.get_view<const Real**,Host>();

    // Coarse grid cols are just copied
    for (int icol=0; icol<ngdofs_src; ++icol) {
      for (int icmp=0; icmp<2; ++icmp) {
        CHECK (tgt_v(2*icol,icmp)==src_v(icol,icmp));
        ok &= catch_capture.lastAssertionPassed();
      }
    }
    // Fine cols are an average of the two cols nearby
    for (int icol=0; icol<ngdofs_src-1; ++icol) {
      for (int icmp=0; icmp<2; ++icmp) {
        avg = (src_v(icol,icmp) + src_v(icol+1,icmp)) / 2;
        CHECK (tgt_v(2*icol+1,icmp)==avg);
        ok &= catch_capture.lastAssertionPassed();
      }
    }
    if (comm.am_i_root()) {
      printf(" -> Checking 2d vectors ......... %s\n",ok ? "PASS" : "FAIL");
    }  
  }

  // Scalar 3D
  {
    if (comm.am_i_root()) {
      printf(" -> Checking 3d scalars .........\n");
    }
    bool ok = true;
    gs3d_src.sync_to_host();
    gs3d_tgt.sync_to_host();

    auto src_v = gs3d_src.get_view<const Real**,Host>();
    auto tgt_v = gs3d_tgt.get_view<const Real**,Host>();

    // Coarse grid cols are just copied
    for (int icol=0; icol<ngdofs_src; ++icol) {
      for (int ilev=0; ilev<nlevs; ++ilev) {
        CHECK (tgt_v(2*icol,ilev)==src_v(icol,ilev));
        ok &= catch_capture.lastAssertionPassed();
      }
    }
    // Fine cols are an average of the two cols nearby
    for (int icol=0; icol<ngdofs_src-1; ++icol) {
      for (int ilev=0; ilev<nlevs; ++ilev) {
        avg = (src_v(icol,ilev) + src_v(icol+1,ilev)) / 2;
        CHECK (tgt_v(2*icol+1,ilev)==avg);
        ok &= catch_capture.lastAssertionPassed();
      }
    }
    if (comm.am_i_root()) {
      printf(" -> Checking 3d scalars ......... %s\n",ok ? "PASS" : "FAIL");
    }
  }

  // Vector 3D
  {
    if (comm.am_i_root()) {
      printf(" -> Checking 3d vectors .........\n");
    }
    bool ok = true;
    gv3d_src.sync_to_host();
    gv3d_tgt.sync_to_host();

    auto src_v = gv3d_src.get_view<const Real***,Host>();
    auto tgt_v = gv3d_tgt.get_view<const Real***,Host>();

    // Coarse grid cols are just copied
    for (int icol=0; icol<ngdofs_src; ++icol) {
      for (int icmp=0; icmp<2; ++icmp) {
        for (int ilev=0; ilev<nlevs; ++ilev) {
          CHECK (tgt_v(2*icol,icmp,ilev)==src_v(icol,icmp,ilev));
          ok &= catch_capture.lastAssertionPassed();
        }
      }
    }
    // Fine cols are an average of the two cols nearby
    for (int icol=0; icol<ngdofs_src-1; ++icol) {
      for (int icmp=0; icmp<2; ++icmp) {
        for (int ilev=0; ilev<nlevs; ++ilev) {
          avg = (src_v(icol,icmp,ilev) + src_v(icol+1,icmp,ilev)) / 2;
          CHECK (tgt_v(2*icol+1,icmp,ilev)==avg);
          ok &= catch_capture.lastAssertionPassed();
        }
      }
    }
    if (comm.am_i_root()) {
      printf(" -> Checking 3d vectors ......... %s\n",ok ? "PASS" : "FAIL");
    }
  }

  // Subfields
  {
    if (comm.am_i_root()) {
      printf(" -> Checking 3d subfields .......\n");
    }
    bool ok = true;
    gbundle_src.sync_to_host();
    gbundle_tgt.sync_to_host();

    for (int icmp=0; icmp<2; ++icmp) {
      auto sf_src = gbundle_src.get_component(icmp);
      auto sf_tgt = gbundle_tgt.get_component(icmp);

      auto src_v = sf_src.get_view<const Real**,Host>();
      auto tgt_v = sf_tgt.get_view<const Real**,Host>();

      // Coarse grid cols are just copied
      for (int icol=0; icol<ngdofs_src; ++icol) {
        for (int ilev=0; ilev<nlevs; ++ilev) {
          CHECK (tgt_v(2*icol,ilev)==src_v(icol,ilev));
          ok &= catch_capture.lastAssertionPassed();
        }
      }
      // Fine cols are an average of the two cols nearby
      for (int icol=0; icol<ngdofs_src-1; ++icol) {
        for (int ilev=0; ilev<nlevs; ++ilev) {
          avg = (src_v(icol,ilev) + src_v(icol+1,ilev)) / 2;
          CHECK (tgt_v(2*icol+1,ilev)==avg);
          ok &= catch_capture.lastAssertionPassed();
        }
      }
    }
    if (comm.am_i_root()) {
      printf(" -> Checking 3d subfields ....... %s\n",ok ? "PASS" : "FAIL");
    }
  }

  // Clean up
  r = nullptr;
  scorpio::finalize_subsystem();
}

// ---------------------------------------------------------------------------
// Helpers shared by the two tests below.
//
// Both tests fill the src fields with an ANALYTIC function of the global col id
// and the non-col indices, rather than random numbers. That way every expected
// tgt value is computable locally, with no MPI gather, so the checks work for
// fields of any rank and at any rank count. The values are small integers, so
// sums are exact and halving is exact: the checks can use ==, independent of
// whether Real is float or double.
// ---------------------------------------------------------------------------

// Value stored at (gid,j,k,l) of a src field
Real ref_val (const int gid, const int j, const int k, const int l) {
  return 8.0*gid + 4.0*j + 2.0*k + 1.0*l;
}

// Visit every entry of a (host) field, calling fn(icol,j,k,l,entry_ref).
// Missing dims are reported as index 0.
template<typename T, typename Fn>
void for_each_entry (const Field& f, Fn&& fn)
{
  const auto& fl = f.get_header().get_identifier().get_layout();
  const int n0 = fl.dim(0);
  const int d1 = fl.rank()>1 ? fl.dim(1) : 1;
  const int d2 = fl.rank()>2 ? fl.dim(2) : 1;
  const int d3 = fl.rank()>3 ? fl.dim(3) : 1;
  switch (fl.rank()) {
    case 1: {
      auto v = f.get_view<T*,Host>();
      for (int i=0; i<n0; ++i) fn(i,0,0,0,v(i));
      break;
    }
    case 2: {
      auto v = f.get_view<T**,Host>();
      for (int i=0; i<n0; ++i) for (int j=0; j<d1; ++j) fn(i,j,0,0,v(i,j));
      break;
    }
    case 3: {
      auto v = f.get_view<T***,Host>();
      for (int i=0; i<n0; ++i) for (int j=0; j<d1; ++j) for (int k=0; k<d2; ++k)
        fn(i,j,k,0,v(i,j,k));
      break;
    }
    case 4: {
      auto v = f.get_view<T****,Host>();
      for (int i=0; i<n0; ++i) for (int j=0; j<d1; ++j) for (int k=0; k<d2; ++k) for (int l=0; l<d3; ++l)
        fn(i,j,k,l,v(i,j,k,l));
      break;
    }
    default:
      EKAT_ERROR_MSG ("Unexpected field rank in refining remapper unit test.\n");
  }
}

// Fill a src field with ref_val, using the grid's global col ids
void fill_src (const Field& f, const AbstractGrid& grid)
{
  using gid_type = AbstractGrid::gid_type;
  auto gids_h = grid.get_dofs_gids().get_view<const gid_type*,Host>();
  for_each_entry<Real>(f,[&](int i, int j, int k, int l, Real& x) {
    x = ref_val(gids_h(i),j,k,l);
  });
  f.sync_to_dev();
}

// The layouts used by both tests. Together they hit every rank case of the
// matvec kernels: rank 1, rank 2 (both packed and unpacked), rank 3 and rank 4.
const std::vector<std::pair<std::string,LayoutType>>& test_layouts ()
{
  static const std::vector<std::pair<std::string,LayoutType>> lts = {
    {"s2d",LayoutType::Scalar2D},   // rank 1
    {"v2d",LayoutType::Vector2D},   // rank 2, no pack size requested
    {"s3d",LayoutType::Scalar3D},   // rank 2, packed
    {"v3d",LayoutType::Vector3D},   // rank 3, packed
    {"t3d",LayoutType::Tensor3D},   // rank 4, packed
  };
  return lts;
}

// This test covers the *masked* refining path, i.e. what an online output stream
// does when it remaps to a finer grid and the fields carry a valid mask.
//
// Mask handling was coarsening-only. When refining, the masked matvec was never even
// dispatched (the check looked for a mask on the ov field, which never has one), so the
// data was remapped WITHOUT the mask but still divided by the remapped mask afterwards,
// inflating every tgt col that draws on a masked src col by 1/sum(w*m). The masked
// kernel itself was also wrong for refining: it used the ov grid row count and a
// src-grid mask, both of which mismatch the refining CRS layout.
TEST_CASE ("refining_remapper_masked") {
  using gid_type = AbstractGrid::gid_type;

  auto& catch_capture = Catch::getResultCapture();

  ekat::Comm comm(MPI_COMM_WORLD);

  scorpio::init_subsystem(comm);

  const int ngdofs_src = 4*comm.size();
  const int ngdofs_tgt = 2*ngdofs_src-1;
  auto filename = "rr_masked_tests_map.np" + std::to_string(comm.size()) + ".nc";
  write_map_file(filename,ngdofs_src);

  // Same grid setup as the unmasked test above
  const int nlevs = std::max(SCREAM_PACK_SIZE,16);
  auto src_grid = create_point_grid("src",ngdofs_src,nlevs,comm,1);
  auto tgt_grid = create_point_grid("tgt",ngdofs_tgt,nlevs,comm,0);
  auto dofs_h = tgt_grid->get_dofs_gids().get_view<gid_type*,Host>();
  for (int i=0; i<tgt_grid->get_num_local_dofs(); ++i) {
    int q = dofs_h[i] / 2;
    if (dofs_h[i] % 2 == 0) {
      dofs_h[i] = q + 1;
    } else {
      dofs_h[i] = ngdofs_src + q + 1;
    }
  }
  tgt_grid->get_dofs_gids().sync_to_dev();

  // Build the remapper WITH mask tracking, like AtmosphereOutput does
  auto r = std::make_shared<HorizontalRemapper>(src_grid,tgt_grid,filename,true);

  // Mask out src col with gid==masked_gid entirely. Note: src gids are 1-based,
  // and gid 2 participates both as a "copy" and in two averages, so this exercises
  // fully-masked and partially-masked tgt cols at once.
  const int masked_gid = 2;
  auto src_gids_h = src_grid->get_dofs_gids().get_view<const gid_type*,Host>();

  std::vector<Field> src_f, tgt_f;
  for (const auto& [name,lt] : test_layouts()) {
    auto fsrc = create_field(name+"_src",lt,*src_grid);
    auto ftgt = create_field(name+"_tgt",lt,*tgt_grid);
    fill_src(fsrc,*src_grid);

    auto& mask = fsrc.create_valid_mask(name+"_mask",Field::MaskInit::Valid);
    // create_valid_mask inits on device, so refresh the host mirror before editing it
    mask.sync_to_host();
    for_each_entry<int>(mask,[&](int i, int, int, int, int& m) {
      if (src_gids_h(i)==masked_gid) m = 0;
    });
    mask.sync_to_dev();

    r->register_field(fsrc,ftgt);
    src_f.push_back(fsrc);
    tgt_f.push_back(ftgt);
  }
  r->registration_ends();

  r->remap_fwd();

  auto tgt_gids_h = tgt_grid->get_dofs_gids().get_view<const gid_type*,Host>();

  for (size_t f=0; f<tgt_f.size(); ++f) {
    const auto& name = test_layouts()[f].first;
    if (comm.am_i_root()) {
      printf(" -> Checking masked %s ..........\n",name.c_str());
    }
    bool ok = true;

    tgt_f[f].sync_to_host();
    for_each_entry<Real>(tgt_f[f],[&](int i, int j, int k, int l, Real& got) {
      const int g = tgt_gids_h(i);
      Real expected;
      if (g<=ngdofs_src) {
        // "copy" col: takes src gid g with weight 1
        expected = g==masked_gid ? constants::fill_value<Real>
                                 : ref_val(g,j,k,l);
      } else {
        // "average" col: 0.5*src(c0) + 0.5*src(c1), renormalized by the mask
        const int c0 = g-ngdofs_src;
        const int c1 = c0+1;
        const bool m0 = c0!=masked_gid;
        const bool m1 = c1!=masked_gid;
        if      (not m0 and not m1) expected = constants::fill_value<Real>;
        else if (not m0)            expected = ref_val(c1,j,k,l);
        else if (not m1)            expected = ref_val(c0,j,k,l);
        else                        expected = (ref_val(c0,j,k,l)+ref_val(c1,j,k,l))/2;
      }
      CHECK (got==expected);
      ok &= catch_capture.lastAssertionPassed();
    });

    // The tgt field must also carry a valid mask, 0 exactly on the fully-masked cols
    REQUIRE (tgt_f[f].has_valid_mask());
    auto& tmask = tgt_f[f].get_valid_mask();
    tmask.sync_to_host();
    for_each_entry<int>(tmask,[&](int i, int, int, int, int& m) {
      const int g = tgt_gids_h(i);
      const bool fully_masked = (g<=ngdofs_src)
                              ? g==masked_gid
                              : (g-ngdofs_src==masked_gid and g-ngdofs_src+1==masked_gid);
      CHECK (m==(fully_masked ? 0 : 1));
      ok &= catch_capture.lastAssertionPassed();
    });

    if (comm.am_i_root()) {
      printf(" -> Checking masked %s .......... %s\n",name.c_str(),ok ? "PASS" : "FAIL");
    }
  }

  // Clean up
  r = nullptr;
  scorpio::finalize_subsystem();
}

// A map file is not obliged to provide weights for every tgt column. Rows with no
// entries used to read weights(beg)/col_lids(beg) with beg==end, which is an
// out-of-bounds read on the last local row (and silently picked up the next row's
// entry for any other row).
TEST_CASE ("refining_remapper_uncovered_tgt_cols") {
  using gid_type = AbstractGrid::gid_type;

  auto& catch_capture = Catch::getResultCapture();

  ekat::Comm comm(MPI_COMM_WORLD);

  scorpio::init_subsystem(comm);

  const int ngdofs_src = 4*comm.size();
  const int ngdofs_tgt = 2*ngdofs_src-1;
  auto filename = "rr_partial_tests_map.np" + std::to_string(comm.size()) + ".nc";
  write_partial_map_file(filename,ngdofs_src);

  const int nlevs = std::max(SCREAM_PACK_SIZE,16);
  auto src_grid = create_point_grid("src",ngdofs_src,nlevs,comm,1);
  auto tgt_grid = create_point_grid("tgt",ngdofs_tgt,nlevs,comm,0);
  auto dofs_h = tgt_grid->get_dofs_gids().get_view<gid_type*,Host>();
  for (int i=0; i<tgt_grid->get_num_local_dofs(); ++i) {
    int q = dofs_h[i] / 2;
    if (dofs_h[i] % 2 == 0) {
      dofs_h[i] = q + 1;
    } else {
      dofs_h[i] = ngdofs_src + q + 1;
    }
  }
  tgt_grid->get_dofs_gids().sync_to_dev();

  auto r = std::make_shared<HorizontalRemapper>(src_grid,tgt_grid,filename);

  std::vector<Field> tgt_f;
  for (const auto& [name,lt] : test_layouts()) {
    auto fsrc = create_field(name+"_src",lt,*src_grid);
    auto ftgt = create_field(name+"_tgt",lt,*tgt_grid);
    fill_src(fsrc,*src_grid);
    // Poison the tgt field, so that "left untouched" cannot masquerade as "set to 0"
    ftgt.deep_copy(-1);
    r->register_field(fsrc,ftgt);
    tgt_f.push_back(ftgt);
  }
  r->registration_ends();

  r->remap_fwd();

  auto tgt_gids_h = tgt_grid->get_dofs_gids().get_view<const gid_type*,Host>();

  for (size_t f=0; f<tgt_f.size(); ++f) {
    const auto& name = test_layouts()[f].first;
    if (comm.am_i_root()) {
      printf(" -> Checking uncovered %s ......\n",name.c_str());
    }
    bool ok = true;

    tgt_f[f].sync_to_host();
    for_each_entry<Real>(tgt_f[f],[&](int i, int j, int k, int l, Real& got) {
      const int g = tgt_gids_h(i);
      // Covered cols are copied; uncovered ones must be 0, NOT the neighbouring
      // row's entry and NOT the poison value we pre-filled
      const Real expected = g<=ngdofs_src ? ref_val(g,j,k,l) : 0;
      CHECK (got==expected);
      ok &= catch_capture.lastAssertionPassed();
    });

    if (comm.am_i_root()) {
      printf(" -> Checking uncovered %s ...... %s\n",name.c_str(),ok ? "PASS" : "FAIL");
    }
  }

  r = nullptr;
  scorpio::finalize_subsystem();
}

} // namespace scream
