#include <catch2/catch.hpp>

#include "share/remap/horizontal_remapper.hpp"
#include "share/grid/point_grid.hpp"
#include "share/scorpio_interface/eamxx_scorpio_interface.hpp"
#include "share/core/eamxx_setup_random_test.hpp"
#include "share/field/field_utils.hpp"
#include "share/util/eamxx_universal_constants.hpp"

namespace scream {

void root_print (const std::string& msg, const ekat::Comm& comm) {
  if (comm.am_i_root()) {
    printf("%s",msg.c_str());
  }
}

// Create a source grid given number of global dofs.
// Dofs are scattered around randomly
std::shared_ptr<AbstractGrid>
build_src_grid(const ekat::Comm& comm, const int ngdofs, int seed)
{
  using gid_type = AbstractGrid::gid_type;
  const int nlevs = 20;

  std::vector<gid_type> all_dofs (ngdofs);
  std::mt19937_64 engine(seed);
  if (comm.am_i_root()) {
    std::iota(all_dofs.data(),all_dofs.data()+all_dofs.size(),1);
    std::shuffle(all_dofs.data(),all_dofs.data()+ngdofs,engine);
  }
  comm.broadcast(all_dofs.data(),ngdofs,comm.root_rank());

  int nldofs = ngdofs / comm.size();
  int remainder = ngdofs % comm.size();
  int offset = nldofs * comm.rank() + std::min(comm.rank(),remainder);
  if (comm.rank()<remainder) {
    ++nldofs;
  }
  auto src_grid = std::make_shared<PointGrid>("src",nldofs,nlevs,comm);

  auto src_dofs = src_grid->get_dofs_gids();
  auto src_dofs_h = src_dofs.get_view<gid_type*,Host>();
  std::copy_n(all_dofs.data()+offset,nldofs,src_dofs_h.data());
  src_dofs.sync_to_dev();

  return src_grid;
}

constexpr int vec_dim = 2;
constexpr int tens_dim1 = 3;
constexpr int tens_dim2 = 4;
Field create_field (const std::string& name, const LayoutType lt, const AbstractGrid& grid, const FieldTag vtag)
{
  using namespace ShortFieldTagsNames;
  const auto& gn = grid.name();
  Field f;
  switch (lt) {
    case LayoutType::Scalar1D:
      f = Field(FieldIdentifier(name,grid.get_vertical_layout(vtag),ekat::units::none,gn)); break;
    case LayoutType::Scalar2D:
      f = Field(FieldIdentifier(name,grid.get_2d_scalar_layout(),ekat::units::none,gn));  break;
    case LayoutType::Vector2D:
      f = Field(FieldIdentifier(name,grid.get_2d_vector_layout(vec_dim),ekat::units::none,gn));  break;
    case LayoutType::Tensor2D:
      f = Field(FieldIdentifier(name,grid.get_2d_tensor_layout({tens_dim1,tens_dim2}),ekat::units::none,gn));  break;
    case LayoutType::Scalar3D:
      f = Field(FieldIdentifier(name,grid.get_3d_scalar_layout(vtag),ekat::units::none,gn));
      f.get_header().get_alloc_properties().request_allocation(SCREAM_PACK_SIZE);
      break;
    case LayoutType::Vector3D:
      f = Field(FieldIdentifier(name,grid.get_3d_vector_layout(vtag,vec_dim),ekat::units::none,gn));
      f.get_header().get_alloc_properties().request_allocation(SCREAM_PACK_SIZE);
      break;
    case LayoutType::Tensor3D:
      f = Field(FieldIdentifier(name,grid.get_3d_tensor_layout(vtag,{tens_dim1,tens_dim2}),ekat::units::none,gn));
      f.get_header().get_alloc_properties().request_allocation(SCREAM_PACK_SIZE);
      break;
    default:
      EKAT_ERROR_MSG ("Invalid layout type for this unit test.\n");
  }
  f.allocate_view();

  return f;
}

Field create_field (const std::string& name, const LayoutType lt, const AbstractGrid& grid, const FieldTag vtag, int seed) {
  auto f = create_field(name,lt,grid,vtag);

  // Use discrete_distribution to get an integer, then use that as exponent for 2^-n.
  // This guarantees numbers that are exactly represented as FP numbers, which ensures
  // the test will produce the expected answer, regardless of how math ops are performed.
  std::vector<Real> values = {1,2,4,8,16,32,64,128,256,512};
  randomize_discrete(f,seed++,values);

  return f;
}

template<typename T>
Field all_gather_field_impl (const Field& f, const ekat::Comm& comm) {
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
  std::vector<T> data_vec(col_size);
  f.sync_to_host();
  for (int pid=0,offset=0; pid<comm.size(); ++pid) {
    T* data;
    int ncols = fl.dims()[0];
    comm.broadcast(&ncols,1,pid);
    for (int icol=0; icol<ncols; ++icol,offset+=col_size) {
      switch (fl.rank()) {
        case 1:
          if (pid==comm.rank()) {
            data = ekat::subview(f.get_view<T*,Host>(),icol).data();
          } else {
            data = data_vec.data();
          }
          break;
        case 2:
          if (pid==comm.rank()) {
            data = ekat::subview(f.get_view<T**,Host>(),icol).data();
          } else {
            data = data_vec.data();
          }
          break;
        case 3:
          if (pid==comm.rank()) {
            data = ekat::subview(f.get_view<T***,Host>(),icol).data();
          } else {
            data = data_vec.data();
          }
          break;
        case 4:
          if (pid==comm.rank()) {
            data = ekat::subview(f.get_view<T****,Host>(),icol).data();
          } else {
            data = data_vec.data();
          }
          break;
        default:
          EKAT_ERROR_MSG (
              "Unexpected rank in RefiningRemapperRMA unit test.\n"
              "  - field name: " + f.name() + "\n");
      }
      comm.broadcast(data,col_size,pid);
      auto gdata = gf.get_internal_view_data<T,Host>()+offset;
      std::copy(data,data+col_size,gdata);
    }
  }
  gf.sync_to_dev();
  return gf;
}

Field all_gather_field (const Field& f, const ekat::Comm& comm) {
  const auto dt = f.data_type();
  if (dt==DataType::RealType) {
    return all_gather_field_impl<Real>(f,comm);
  } else {
    return all_gather_field_impl<int>(f,comm);
  }
}

// Helper function to create a remap file
void create_remap_file(const std::string& filename, const int ngdofs_tgt)
{
  const int ngdofs_src = ngdofs_tgt + 1;
  const int nnz = 2*ngdofs_tgt;

  scorpio::register_file(filename, scorpio::FileMode::Write);

  scorpio::define_dim(filename,"n_a", ngdofs_src);
  scorpio::define_dim(filename,"n_b", ngdofs_tgt);
  scorpio::define_dim(filename,"n_s", nnz);

  scorpio::define_var(filename,"col",{"n_s"},"int");
  scorpio::define_var(filename,"row",{"n_s"},"int");
  scorpio::define_var(filename,"S"  ,{"n_s"},"double");

  scorpio::enddef(filename);

  std::vector<int> col(nnz), row(nnz);
  std::vector<double> S(nnz,0.5);
  const int gid_base = 1;
  for (int i=0; i<ngdofs_tgt; ++i) {
    row[2*i]   = gid_base + i;
    row[2*i+1] = gid_base + i;
    col[2*i]   = gid_base + i;
    col[2*i+1] = gid_base + i+1;
  }

  scorpio::write_var(filename,"row",row.data());
  scorpio::write_var(filename,"col",col.data());
  scorpio::write_var(filename,"S",    S.data());

  scorpio::release_file(filename);
}

TEST_CASE("coarsening_remap")
{
  using namespace ShortFieldTagsNames;
  auto& catch_capture = Catch::getResultCapture();

  // This is a simple test to just make sure the coarsening remapper works
  // when the map itself has more remap triplets than the size of the
  // source and target grid.  This is typical in monotone remappers from
  // fine to coarse meshes.

  // -------------------------------------- //
  //           Init MPI and PIO             //
  // -------------------------------------- //

  ekat::Comm comm(MPI_COMM_WORLD);

  root_print ("\n +---------------------------------+\n",comm);
  root_print (" |   Testing coarsening remapper   |\n",comm);
  root_print (" +---------------------------------+\n\n",comm);

  scorpio::init_subsystem(comm);
  int seed = get_random_test_seed(&comm);

  // -------------------------------------- //
  //           Create a map file            //
  // -------------------------------------- //

  std::string filename = "cr_tests_map." + std::to_string(comm.size()) + ".nc";

  const int nldofs_tgt = 3;
  const int ngdofs_tgt = nldofs_tgt*comm.size();
  create_remap_file(filename, ngdofs_tgt);

  // -------------------------------------- //
  //      Build src grid and remapper       //
  // -------------------------------------- //

  const int ngdofs_src = ngdofs_tgt+1;
  auto src_grid = build_src_grid(comm, ngdofs_src, seed);
  auto remap = std::make_shared<HorizontalRemapper>(src_grid,filename);

  // -------------------------------------- //
  //      Create src/tgt grid fields        //
  // -------------------------------------- //

  // The other test checks remapping for fields of multiple dimensions.
  // Here we will simplify and just remap a simple 2D horizontal field.
  auto tgt_grid = remap->get_tgt_grid();

  auto src_s1d   = create_field("s1d",  LayoutType::Scalar1D, *src_grid, LEV,  seed+1);
  auto src_s2d   = create_field("s2d",  LayoutType::Scalar2D, *src_grid, ILEV, seed+2);
  auto src_v2d   = create_field("v2d",  LayoutType::Vector2D, *src_grid, ILEV, seed+3);
  auto src_t2d   = create_field("t2d",  LayoutType::Tensor2D, *src_grid, ILEV, seed+4);
  auto src_s3d_m = create_field("s3d_m",LayoutType::Scalar3D, *src_grid, LEV,  seed+5);
  auto src_s3d_i = create_field("s3d_i",LayoutType::Scalar3D, *src_grid, ILEV, seed+6);
  auto src_v3d_m = create_field("v3d_m",LayoutType::Vector3D, *src_grid, LEV,  seed+7);
  auto src_v3d_i = create_field("v3d_i",LayoutType::Vector3D, *src_grid, ILEV, seed+8);
  auto src_t3d_m = create_field("t3d_m",LayoutType::Tensor3D, *src_grid, LEV,  seed+9);
  auto src_t3d_i = create_field("t3d_i",LayoutType::Tensor3D, *src_grid, ILEV, seed+10);

  auto tgt_s1d   = create_field("s1d",  LayoutType::Scalar1D, *tgt_grid, LEV);
  auto tgt_s2d   = create_field("s2d",  LayoutType::Scalar2D, *tgt_grid, ILEV);
  auto tgt_v2d   = create_field("v2d",  LayoutType::Vector2D, *tgt_grid, ILEV);
  auto tgt_t2d   = create_field("t2d",  LayoutType::Tensor2D, *tgt_grid, ILEV);
  auto tgt_s3d_m = create_field("s3d_m",LayoutType::Scalar3D, *tgt_grid, LEV );
  auto tgt_s3d_i = create_field("s3d_i",LayoutType::Scalar3D, *tgt_grid, ILEV);
  auto tgt_v3d_m = create_field("v3d_m",LayoutType::Vector3D, *tgt_grid, LEV );
  auto tgt_v3d_i = create_field("v3d_i",LayoutType::Vector3D, *tgt_grid, ILEV);
  auto tgt_t3d_m = create_field("t3d_m",LayoutType::Tensor3D, *tgt_grid, LEV );
  auto tgt_t3d_i = create_field("t3d_i",LayoutType::Tensor3D, *tgt_grid, ILEV);

  std::vector<Field> src_f = {src_s1d,src_s2d,src_v2d,src_t2d,src_s3d_m,src_s3d_i,src_v3d_m,src_v3d_i,src_t3d_m,src_t3d_i};
  std::vector<Field> tgt_f = {tgt_s1d,tgt_s2d,tgt_v2d,tgt_t2d,tgt_s3d_m,tgt_s3d_i,tgt_v3d_m,tgt_v3d_i,tgt_t3d_m,tgt_t3d_i};

  // -------------------------------------- //
  //     Register fields in the remapper    //
  // -------------------------------------- //

  for (size_t i=0; i<tgt_f.size(); ++i) {
    remap->register_field(src_f[i],tgt_f[i]);
  }
  remap->registration_ends();

  // -------------------------------------- //
  //          Check remapped fields         //
  // -------------------------------------- //

  Real w = 0.5;
  auto gids_tgt = all_gather_field(tgt_grid->get_dofs_gids().clone(CloneFlags::CopyData),comm); // Need clone to be able to extract writable
  auto gids_src = all_gather_field(src_grid->get_dofs_gids().clone(CloneFlags::CopyData),comm); // pointers to pass to MPI's broadcast
  auto gids_src_v = gids_src.get_view<const AbstractGrid::gid_type*,Host>();
  auto gids_tgt_v = gids_tgt.get_view<const AbstractGrid::gid_type*,Host>();

  auto gid2lid = [&](const int gid, const auto gids_v) {
    auto data = gids_v.data();
    auto it = std::find(data,data+gids_v.size(),gid);
    return std::distance(data,it);
  };
  constexpr int nruns = 5;
  for (int irun=0; irun<nruns; ++irun) {
    root_print (" -> Run " + std::to_string(irun) + "\n",comm);
    remap->remap_fwd();

    // Recall, tgt gid K should be the avg of local src_gids
    for (size_t ifield=0; ifield<tgt_f.size(); ++ifield) {
      auto gsrc = all_gather_field(src_f[ifield],comm);
      auto gtgt = all_gather_field(tgt_f[ifield],comm);

      const auto& l = gsrc.get_header().get_identifier().get_layout();
      const auto ls = l.to_string();
      std::string dots (30-ls.size(),'.');
      auto msg = "   -> Checking field with layout " + ls + " " + dots;
      root_print (msg + "\n",comm);
      bool ok = true;
      switch (l.type()) {
        case LayoutType::Scalar1D:
        {
          CHECK ( views_are_equal(gsrc,gtgt) );
          ok &= catch_capture.lastAssertionPassed();
        } break;
        case LayoutType::Scalar2D:
        {
          const auto v_src = gsrc.get_view<const Real*,Host>();
          const auto v_tgt = gtgt.get_view<const Real*,Host>();
          for (int idof=0; idof<ngdofs_tgt; ++idof) {
            Real expected = 0;
            auto gdof = gids_tgt_v(idof);
            for (int j=0; j<2; ++j) {
              auto src_gcol = gdof + j;
              auto src_lcol = gid2lid(src_gcol,gids_src_v);
              expected += w*v_src(src_lcol);
            }
            CHECK ( v_tgt(idof)== expected );
            ok &= catch_capture.lastAssertionPassed();
          }
        } break;
        case LayoutType::Vector2D:
        {
          const auto v_src = gsrc.get_view<const Real**,Host>();
          const auto v_tgt = gtgt.get_view<const Real**,Host>();
          for (int idof=0; idof<ngdofs_tgt; ++idof) {
            for (int icmp=0; icmp<vec_dim; ++icmp) {
              Real expected = 0;
              auto gdof = gids_tgt_v(idof);
              for (int j=0; j<2; ++j) {
                auto src_gcol = gdof + j;
                auto src_lcol = gid2lid(src_gcol,gids_src_v);
                expected += w*v_src(src_lcol,icmp);
              }
              CHECK ( v_tgt(idof,icmp)== expected );
              ok &= catch_capture.lastAssertionPassed();
            }
          }
        } break;
        case LayoutType::Tensor2D:
        {
          const auto v_src = gsrc.get_view<const Real***,Host>();
          const auto v_tgt = gtgt.get_view<const Real***,Host>();
          for (int idof=0; idof<ngdofs_tgt; ++idof) {
            for (int icmp=0; icmp<vec_dim; ++icmp) {
              for (int jcmp=0; jcmp<vec_dim; ++jcmp) {
                Real expected = 0;
                auto gdof = gids_tgt_v(idof);
                for (int j=0; j<2; ++j) {
                  auto src_gcol = gdof + j;
                  auto src_lcol = gid2lid(src_gcol,gids_src_v);
                  expected += w*v_src(src_lcol,icmp,jcmp);
                }
                CHECK ( v_tgt(idof,icmp,jcmp)== expected );
                ok &= catch_capture.lastAssertionPassed();
              }
            }
          }
        } break;
        case LayoutType::Scalar3D:
        {
          const auto v_src = gsrc.get_view<const Real**,Host>();
          const auto v_tgt = gtgt.get_view<const Real**,Host>();
          auto f_nlevs = gsrc.get_header().get_identifier().get_layout().dims().back();
          for (int idof=0; idof<ngdofs_tgt; ++idof) {
            for (int ilev=0; ilev<f_nlevs; ++ilev) {
              Real expected = 0;
              auto gdof = gids_tgt_v(idof);
              for (int j=0; j<2; ++j) {
                auto src_gcol = gdof + j;
                auto src_lcol = gid2lid(src_gcol,gids_src_v);
                expected += w*v_src(src_lcol,ilev);
              }
              CHECK ( v_tgt(idof,ilev)== expected );
              ok &= catch_capture.lastAssertionPassed();
            }
          }
        } break;
        case LayoutType::Vector3D:
        {
          const auto v_src = gsrc.get_view<const Real***,Host>();
          const auto v_tgt = gtgt.get_view<const Real***,Host>();
          auto f_nlevs = gsrc.get_header().get_identifier().get_layout().dims().back();
          for (int idof=0; idof<ngdofs_tgt; ++idof) {
            for (int icmp=0; icmp<vec_dim; ++icmp) {
              for (int ilev=0; ilev<f_nlevs; ++ilev) {
                Real expected = 0;
                auto gdof = gids_tgt_v(idof);
                for (int j=0; j<2; ++j) {
                  auto src_gcol = gdof + j;
                  auto src_lcol = gid2lid(src_gcol,gids_src_v);
                  expected += w*v_src(src_lcol,icmp,ilev);
                }
                CHECK ( v_tgt(idof,icmp,ilev)== expected );
                ok &= catch_capture.lastAssertionPassed();
              }
            }
          }
        } break;
        case LayoutType::Tensor3D:
        {
          const auto v_src = gsrc.get_view<const Real****,Host>();
          const auto v_tgt = gtgt.get_view<const Real****,Host>();
          auto f_nlevs = gsrc.get_header().get_identifier().get_layout().dims().back();
          for (int idof=0; idof<ngdofs_tgt; ++idof) {
            for (int icmp=0; icmp<tens_dim1; ++icmp) {
              for (int jcmp=0; jcmp<tens_dim2; ++jcmp) {
                for (int ilev=0; ilev<f_nlevs; ++ilev) {
                  Real expected = 0;
                  auto gdof = gids_tgt_v(idof);
                  for (int j=0; j<2; ++j) {
                    auto src_gcol = gdof + j;
                    auto src_lcol = gid2lid(src_gcol,gids_src_v);
                    expected += w*v_src(src_lcol,icmp,jcmp,ilev);
                  }
                  CHECK ( v_tgt(idof,icmp,jcmp,ilev)== expected );
                  ok &= catch_capture.lastAssertionPassed();
                }
              }
            }
          }
        } break;
        default:
          EKAT_ERROR_MSG ("Unexpected layout.\n");
      }
      root_print (msg + (ok ? "PASS" : "FAIL") + "\n",comm);
    }
  }

  // Clean up scorpio stuff
  scorpio::finalize_subsystem();
}

// ---------------------------------------------------------------------------
// Masked coarsening.
//
// Mask tracking was originally a coarsening-only feature, so this direction was the
// one that worked. It is covered end-to-end by io_remap_test, but had no direct unit
// test, which left it exposed when the masked matvec was reworked to serve both
// directions. This pins it down: the dispatch decision, the choice of mask field, and
// the mask renormalization must all stay correct when coarsening.
// ---------------------------------------------------------------------------

// Value stored at (gid,j,k,l) of a src field. Small integers, so sums are exact and
// halving is exact: the checks below can use == in single or double precision.
Real cr_ref_val (const int gid, const int j, const int k, const int l) {
  return 8.0*gid + 4.0*j + 2.0*k + 1.0*l;
}

// Visit every entry of a (host) field, calling fn(icol,j,k,l,entry_ref)
template<typename T, typename Fn>
void cr_for_each_entry (const Field& f, Fn&& fn)
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
      EKAT_ERROR_MSG ("Unexpected field rank in coarsening remapper unit test.\n");
  }
}

TEST_CASE("coarsening_remap_masked")
{
  using namespace ShortFieldTagsNames;
  using gid_type = AbstractGrid::gid_type;
  auto& catch_capture = Catch::getResultCapture();

  ekat::Comm comm(MPI_COMM_WORLD);
  int seed = get_random_test_seed(&comm);
  scorpio::init_subsystem(comm);

  std::string filename = "cr_masked_tests_map." + std::to_string(comm.size()) + ".nc";

  // 4 tgt cols per rank, so that the two masked src cols below always exist
  const int nldofs_tgt = 4;
  const int ngdofs_tgt = nldofs_tgt*comm.size();
  create_remap_file(filename, ngdofs_tgt);

  const int ngdofs_src = ngdofs_tgt+1;
  auto src_grid = build_src_grid(comm, ngdofs_src, seed);
  auto remap = std::make_shared<HorizontalRemapper>(src_grid,filename,true);
  auto tgt_grid = remap->get_tgt_grid();

  // The map is tgt(g) = 0.5*src(g) + 0.5*src(g+1). Masking two ADJACENT src cols means
  // tgt col M has both its contributors masked, exercising the fill-value path, while
  // tgt cols M-1 and M+1 have exactly one masked contributor, exercising renormalization.
  const int M = 3;
  REQUIRE (M+1 <= ngdofs_src);

  auto src_gids_h = src_grid->get_dofs_gids().get_view<const gid_type*,Host>();

  // Every rank case of the matvec kernels: rank 1, rank 2 (unpacked and packed),
  // rank 3 and rank 4.
  const std::vector<std::pair<std::string,std::pair<LayoutType,FieldTag>>> layouts = {
    {"s2d",{LayoutType::Scalar2D,ILEV}},   // rank 1
    {"v2d",{LayoutType::Vector2D,ILEV}},   // rank 2, unpacked
    {"s3d",{LayoutType::Scalar3D,LEV }},   // rank 2, packed
    {"v3d",{LayoutType::Vector3D,LEV }},   // rank 3
    {"t3d",{LayoutType::Tensor3D,LEV }},   // rank 4
  };

  std::vector<Field> tgt_f;
  for (const auto& [name,cfg] : layouts) {
    auto fsrc = create_field(name+"_src",cfg.first,*src_grid,cfg.second);
    auto ftgt = create_field(name+"_tgt",cfg.first,*tgt_grid,cfg.second);

    cr_for_each_entry<Real>(fsrc,[&](int i, int j, int k, int l, Real& x) {
      x = cr_ref_val(src_gids_h(i),j,k,l);
    });
    fsrc.sync_to_dev();

    auto& mask = fsrc.create_valid_mask(name+"_mask",Field::MaskInit::Valid);
    mask.sync_to_host();
    cr_for_each_entry<int>(mask,[&](int i, int, int, int, int& m) {
      if (src_gids_h(i)==M or src_gids_h(i)==M+1) m = 0;
    });
    mask.sync_to_dev();

    remap->register_field(fsrc,ftgt);
    tgt_f.push_back(ftgt);
  }
  remap->registration_ends();

  remap->remap_fwd();

  auto tgt_gids_h = tgt_grid->get_dofs_gids().get_view<const gid_type*,Host>();

  for (size_t f=0; f<tgt_f.size(); ++f) {
    const auto& name = layouts[f].first;
    root_print(" -> Checking masked coarsening " + name + " ...\n",comm);
    bool ok = true;

    tgt_f[f].sync_to_host();
    cr_for_each_entry<Real>(tgt_f[f],[&](int i, int j, int k, int l, Real& got) {
      const int g  = tgt_gids_h(i);
      const int c0 = g;
      const int c1 = g+1;
      auto valid = [&](int c) { return c!=M and c!=M+1; };
      Real expected;
      if      (valid(c0) and valid(c1)) expected = (cr_ref_val(c0,j,k,l)+cr_ref_val(c1,j,k,l))/2;
      else if (valid(c0))               expected = cr_ref_val(c0,j,k,l);
      else if (valid(c1))               expected = cr_ref_val(c1,j,k,l);
      else                              expected = constants::fill_value<Real>;
      CHECK (got==expected);
      ok &= catch_capture.lastAssertionPassed();
    });

    // tgt mask must be 0 exactly on the cols with no valid contributor
    REQUIRE (tgt_f[f].has_valid_mask());
    auto& tmask = tgt_f[f].get_valid_mask();
    tmask.sync_to_host();
    cr_for_each_entry<int>(tmask,[&](int i, int, int, int, int& m) {
      const int g = tgt_gids_h(i);
      const bool fully_masked = (g==M or g==M+1) and (g+1==M or g+1==M+1);
      CHECK (m==(fully_masked ? 0 : 1));
      ok &= catch_capture.lastAssertionPassed();
    });

    root_print(" -> Checking masked coarsening " + name + " ... " + (ok ? "PASS" : "FAIL") + "\n",comm);
  }

  remap = nullptr;
  scorpio::finalize_subsystem();
}

} // namespace scream
