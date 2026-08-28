#include <catch2/catch.hpp>

#include "share/remap/vertical_remapper.hpp"
#include "share/grid/point_grid.hpp"
#include "share/scorpio_interface/eamxx_scorpio_interface.hpp"
#include "share/util/eamxx_timing.hpp"
#include "share/field/field_utils.hpp"

#include <cmath>

namespace scream {

constexpr int vec_dim = 3;
constexpr auto P0 = VerticalRemapper::P0;
constexpr auto Mask = VerticalRemapper::Mask;
constexpr auto Top = VerticalRemapper::Top;
constexpr auto Bot = VerticalRemapper::Bot;
constexpr Real fill_val = constants::fill_value<Real>;

template<typename... Args>
void print (const std::string& fmt, const ekat::Comm& comm, Args&&... args) {
  if (comm.am_i_root()) {
    printf(fmt.c_str(),std::forward<Args>(args)...);
  }
}
// Overload for when there are no additional arguments
void print(const std::string& fmt, const ekat::Comm& comm) {
  if (comm.am_i_root()) {
    printf("%s",fmt.c_str());
  }
}

// Helper function to create a grid given the number of dof's and a comm group.
std::shared_ptr<AbstractGrid>
build_grid(const ekat::Comm& comm, const int nldofs, const int nlevs)
{
  using gid_type = AbstractGrid::gid_type;

  auto grid = std::make_shared<PointGrid>("src",nldofs,nlevs,comm);

  auto dofs = grid->get_dofs_gids();
  auto dofs_h = dofs.get_view<gid_type*,Host>();
  std::iota(dofs_h.data(),dofs_h.data()+nldofs,nldofs*comm.rank());
  dofs.sync_to_dev();

  return grid;
}

// Helper function to create fields
Field
create_field(const std::string& name,
             const std::shared_ptr<const AbstractGrid>& grid,
             const bool create_valid_mask, const bool twod,
             const bool vec, const FieldTag vtag = FieldTag::Invalid, const int ps = 1)
{
  using namespace ShortFieldTagsNames;
  constexpr int vec_dim = 3;
  auto fl = twod
          ? (vec ? grid->get_2d_vector_layout (vec_dim)
                 : grid->get_2d_scalar_layout ())
          : (vec ? grid->get_3d_vector_layout (vtag,vec_dim)
                 : grid->get_3d_scalar_layout (vtag));
  FieldIdentifier fid(name,fl,ekat::units::none,grid->name());
  Field f(fid);
  f.get_header().get_alloc_properties().request_allocation(ps);
  f.allocate_view();

  if (create_valid_mask) {
    f.create_valid_mask(Field::MaskInit::Valid);
  }
  return f;
}

// Helper function to set variable data
Real data_func(const int col, const int vec, const Real pres) {
  // Using a linear function dependent on 
  //   - col, the horizontal column,
  //   - vec, the vector dimension, and
  //   - pres, the current pressure
  // Should ensure that the interpolated values match exactly, since vertical interp is also a linear interpolator.
  // Note, we don't use the level, because here the vertical interpolation is over pressure, so it represents the level.
  return (col+1)*pres + vec*100.0;
}

void compute_field (const Field& f, const Field& p)
{
  Field::view_host_t<const Real*> p1d;
  Field::view_host_t<const Real**> p2d;
  bool rank1 = p.rank()==1;
  const auto& l = f.get_header().get_identifier().get_layout();
  const int ncols = l.dims().front();
  const int nlevs = l.dims().back();
  if (rank1) {
    p1d = p.get_view<const Real*,Host>();
  } else {
    p2d = p.get_view<const Real**,Host>();
  }

  // Grab correct pressure (1d or 2d)
  auto pval = [&](int i, int k) {
    if (rank1) return p1d(k);
    else       return p2d(i,k);
  };

  switch (l.type()) {
    case LayoutType::Scalar2D:
    {
      const auto v = f.get_view<Real*,Host>();
      for (int i=0; i<ncols; ++i) {
        v(i) = i+1;
      }
    } break;
    case LayoutType::Vector2D:
    {
      const auto v = f.get_view<Real**,Host>();
      for (int i=0; i<ncols; ++i) {
        for (int j=0; j<vec_dim; ++j) {
          v(i,j) = i+1 + ncols*(j+1);
      }}
    } break;
    case LayoutType::Scalar3D:
    {
      const auto v = f.get_view<Real**,Host>();
      for (int i=0; i<ncols; ++i) {
        for (int j=0; j<nlevs; ++j) {
          v(i,j) = data_func(i,0,pval(i,j));
      }}
    } break;
    case LayoutType::Vector3D:
    {
      const auto v = f.get_view<Real***,Host>();
      for (int i=0; i<ncols; ++i) {
        for (int j=0; j<vec_dim; ++j) {
          for (int k=0; k<nlevs; ++k) {
            v(i,j,k) = data_func(i,j,pval(i,k));
      }}}
    } break;
    default:
      EKAT_ERROR_MSG ("Unexpected layout.\n");
  }
  f.sync_to_dev();
}

void extrapolate (const Field& p_src, const Field& p_tgt, const Field& f,
                  const VerticalRemapper::ExtrapType etype_top,
                  const VerticalRemapper::ExtrapType etype_bot)
{
  Field::view_host_t<const Real*>  p1d_src,p1d_tgt;
  Field::view_host_t<const Real**> p2d_src,p2d_tgt;
  if (p_src.rank()==1) {
    p1d_src = p_src.get_view<const Real*,Host>();
  } else {
    p2d_src = p_src.get_view<const Real**,Host>();
  }
  if (p_tgt.rank()==1) {
    p1d_tgt = p_tgt.get_view<const Real*,Host>();
  } else {
    p2d_tgt = p_tgt.get_view<const Real**,Host>();
  }

  auto pval = [&](auto p1d, auto p2d, int i, int k, int rank) {
    if (rank==1) return p1d(k);
    else         return p2d(i,k);
  };

  const auto& l = f.get_header().get_identifier().get_layout();
  const int ncols = l.dims().front();
  const int nlevs = l.dims().back();
  const int nlevs_src = p_src.get_header().get_identifier().get_layout().dims().back();
  auto mask = f.get_valid_mask();

  switch (l.type()) {
    case LayoutType::Scalar2D: break;
    case LayoutType::Vector2D: break;
    case LayoutType::Scalar3D:
    {
      const auto v = f.get_view<Real**,Host>();
      const auto m = mask.get_view<int**,Host>();

      for (int i=0; i<ncols; ++i) {
        auto pmin = pval(p1d_src,p2d_src,i,0,p_src.rank());
        auto pmax = pval(p1d_src,p2d_src,i,nlevs_src-1,p_src.rank());
        for (int j=0; j<nlevs; ++j) {
          auto p = pval(p1d_tgt,p2d_tgt,i,j,p_tgt.rank());
          if (p>pmax) {
            if (etype_bot==Mask) {
              v(i,j) = fill_val;
              m(i,j) = 0;
            } else {
              v(i,j) = data_func(i,0,pmax);
            }
          } else if (p<pmin) {
            if (etype_top==Mask) {
              v(i,j) = fill_val;
              m(i,j) = 0;
            } else {
              v(i,j) = data_func(i,0,pmin);
            }
          }
      }}
    } break;
    case LayoutType::Vector3D:
    {
      const auto v = f.get_view<Real***,Host>();
      const auto m = mask.get_view<int***,Host>();

      for (int i=0; i<ncols; ++i) {
        auto pmin = pval(p1d_src,p2d_src,i,0,p_src.rank());
        auto pmax = pval(p1d_src,p2d_src,i,nlevs_src-1,p_src.rank());
        for (int j=0; j<vec_dim; ++j) {
          for (int k=0; k<nlevs; ++k) {
            auto p = pval(p1d_tgt,p2d_tgt,i,k,p_tgt.rank());
            if (p>pmax) {
              if (etype_bot==Mask) {
                v(i,j,k) = fill_val;
                m(i,j,k) = 1;
              } else {
                v(i,j,k) = data_func(i,j,pmax);
              }
            } else if (p<pmin) {
              if (etype_top==Mask) {
                v(i,j,k) = fill_val;
                m(i,j,k) = 1;
              } else {
                v(i,j,k) = data_func(i,j,pmin);
              }
            }
      }}}
    } break;
    default:
      EKAT_ERROR_MSG ("Unexpected layout.\n");
  }
  f.sync_to_dev();

  mask.sync_to_dev();
}

// Helper function to create a remap file
void create_remap_file(const std::string& filename, const int nlevs, const std::vector<std::int64_t>& dofs_p, const std::vector<Real>& p_tgt) 
{
  scorpio::register_file(filename, scorpio::FileMode::Write);
  scorpio::define_dim(filename,"lev",nlevs);
  scorpio::define_var(filename,"p_levs",{"lev"},"real");
  scorpio::enddef(filename);

  scorpio::write_var(filename,"p_levs",p_tgt.data());

  scorpio::release_file(filename);
}

TEST_CASE ("create_tgt_grid") {

  // -------------------------------------- //
  //           Init MPI and PIO             //
  // -------------------------------------- //

  ekat::Comm comm(MPI_COMM_WORLD);

  print ("Testing retrieval of tgt grid from map file ...\n",comm);

  scorpio::init_subsystem(comm);

  // -------------------------------------- //
  //           Set grid/map sizes           //
  // -------------------------------------- //

  const int nlevs_src  = 2*SCREAM_PACK_SIZE + 2;  // Make sure we check what happens when the vertical extent is a little larger than the max PACK SIZE
  const int nlevs_tgt  = nlevs_src/2;
  const int nldofs = 1;

  // -------------------------------------- //
  //           Create a map file            //
  // -------------------------------------- //

  print (" -> creating map file ...\n",comm);


  std::string filename = "vertical_map_file_np" + std::to_string(comm.size()) + ".nc";

  // Create target pressure levels to be remapped onto
  const Real ptop_tgt = 1;
  const Real pbot_tgt = nlevs_src-2;
  const Real dp_tgt   = (pbot_tgt-ptop_tgt)/(nlevs_tgt-1);
  std::vector<std::int64_t> dofs_p(nlevs_tgt);
  std::iota(dofs_p.begin(),dofs_p.end(),0);
  std::vector<Real> p_tgt;
  for (int k=0; k<nlevs_tgt; ++k) {
    p_tgt.push_back(ptop_tgt + dp_tgt*k);
  }

  create_remap_file(filename, nlevs_tgt, dofs_p, p_tgt);
  print (" -> creating map file ... done!\n",comm);

  // -------------------------------------- //
  //             Build src grid             //
  // -------------------------------------- //

  print (" -> creating src grid ...\n",comm);
  auto src_grid = build_grid(comm, nldofs, nlevs_src);
  print (" -> creating src grid ... done!\n",comm);

  // -------------------------------------- //
  //           Retrieve tgt grid            //
  // -------------------------------------- //

  print (" -> retreiving tgt grid ...\n",comm);
  auto tgt_grid = VerticalRemapper::create_tgt_grid (src_grid,filename);
  print (" -> retreiving tgt grid ... done!\n",comm);

  // -------------------------------------- //
  //             Check tgt grid             //
  // -------------------------------------- //

  print (" -> checking tgt grid ...\n",comm);
  REQUIRE (tgt_grid->get_num_local_dofs()==src_grid->get_num_local_dofs());
  REQUIRE (tgt_grid->get_num_vertical_levels()==nlevs_tgt);
  REQUIRE (tgt_grid->has_geometry_data("p_levs"));
  auto p_levs = tgt_grid->get_geometry_data("p_levs");
  auto p_levs_v = p_levs.get_view<const Real*,Host>();
  for (int k=0; k<nlevs_tgt; ++k) {
    REQUIRE (p_tgt[k]==p_levs_v[k]);
  }

  print (" -> checking tgt grid ... done!\n",comm);

  // Clean up scorpio stuff
  scorpio::finalize_subsystem();

  print ("Testing retrieval of tgt grid from map file ...\n",comm);
}

TEST_CASE ("vertical_remapper") {
  using namespace ShortFieldTagsNames;
  // -------------------------------------- //
  //           Init MPI and PIO             //
  // -------------------------------------- //

  ekat::Comm comm(MPI_COMM_WORLD);

  print ("Testing vertical remapper ...\n",comm);

  scorpio::init_subsystem(comm);

  // -------------------------------------- //
  //           Set grid/map sizes           //
  // -------------------------------------- //

  const int nlevs_src  = 2*SCREAM_PACK_SIZE + 2;  // Make sure we check what happens when the vertical extent is a little larger than the max PACK SIZE
  const int nldofs = 1;

  // -------------------------------------- //
  //             Build src grid             //
  // -------------------------------------- //

  print (" -> creating src grid ...\n",comm);
  auto src_grid = build_grid(comm, nldofs, nlevs_src);
  print ("      nlevs src: %d\n",comm,nlevs_src);
  print (" -> creating src grid ...done!\n",comm);

  // Tgt grid must have same 2d layout as src grid
  REQUIRE_THROWS (std::make_shared<VerticalRemapper>(src_grid,build_grid(comm,nldofs+1,nlevs_src)));

  // Helper lambda, to create p_int profile. If it is a 3d field, make same profile on each col
  auto create_pint = [&](const auto& grid, const bool one_d, const Real ptop, const Real pbot) {
    using namespace ShortFieldTagsNames;
    auto tag = grid->get_vkind()==AbstractGrid::VKind::Model ? ILEV : LEVP;
    auto layout = one_d ? grid->get_vertical_layout(tag)
                        : grid->get_3d_scalar_layout(tag);
    FieldIdentifier fid("p_int",layout,ekat::units::Pa,grid->name());
    Field pint (fid);
    pint.get_header().get_alloc_properties().request_allocation(SCREAM_PACK_SIZE);
    pint.allocate_view();

    bool p_grid = grid->get_vkind()==AbstractGrid::VKind::Pressure;
    int num_intervals = grid->get_num_vertical_levels();
    if (p_grid)
      --num_intervals;
    const Real dp = (pbot-ptop)/num_intervals;

    if (one_d) {
      auto pv = pint.get_view<Real*,Host>();
      pv(num_intervals) = pbot;
      for (int k=num_intervals; k>0; --k) {
        pv(k-1) = pv(k) - dp;
      }
    } else {
      auto pv = pint.get_view<Real**,Host>();
      for (int i=0; i<nldofs; ++i) {
        pv(i,num_intervals) = pbot;
        for (int k=num_intervals; k>0; --k) {
          pv(i,k-1) = pv(i,k) - dp;
        }
      }
    }
    pint.sync_to_dev();
    return pint;
  };

  // Helper lambda to create pmid from pint
  auto create_pmid = [&](const Field& pint) {
    using namespace ShortFieldTagsNames;
    auto fid_int = pint.get_header().get_identifier();
    auto layout = fid_int.get_layout();
    if (layout.tags().back()==LEVP) {
      return pint;
    }
    int nlevs = layout.dims().back()-1;
    layout.strip_dim(ILEV);
    layout.append_dim(LEV,nlevs);
    FieldIdentifier fid("p_mid",layout,ekat::units::Pa,fid_int.get_grid_name());
    Field pmid(fid);
    pmid.get_header().get_alloc_properties().request_allocation(SCREAM_PACK_SIZE);
    pmid.allocate_view();
    if (pmid.rank()==1) {
      auto pint_v = pint.get_view<const Real*,Host>();
      auto pmid_v = pmid.get_view<      Real*,Host>();
      for (int k=0; k<nlevs; ++k) {
        pmid_v(k) = 0.5*(pint_v(k) + pint_v(k+1));
      }
    } else {
      auto pint_v = pint.get_view<const Real**,Host>();
      auto pmid_v = pmid.get_view<      Real**,Host>();
      for (int i=0; i<nldofs; ++i) {
        for (int k=0; k<nlevs; ++k) {
          pmid_v(i,k) = 0.5*(pint_v(i,k) + pint_v(i,k+1));
        }
      }
    }
    pmid.sync_to_dev();
    return pmid;
  };

  const Real ptop_src = 50;
  const Real pbot_src = 1000;

  // Test tgt grid with 2x and 0.5x as many levels as src grid
  for (int nlevs_tgt : {nlevs_src/2, 2*nlevs_src}) {
    for (bool src_1d : {true, false}) {
      for (bool tgt_1d : {true, false}) {
        for (auto etype_top : {P0, Mask}) {
          for (auto etype_bot : {P0, Mask}) {
            print ("************************************************\n",comm);
            print ("      nlevs tgt: %d\n",comm,nlevs_tgt);
            print ("      src pressure is 1d: %s\n",comm,src_1d ? "true" : "false");
            print ("      tgt pressure is 1d: %s\n",comm,tgt_1d ? "true" : "false");
            print ("      extrap type at top: %s\n",comm,etype_top==P0 ? "p0" : "masked");
            print ("      extrap type at bot: %s\n",comm,etype_bot==P0 ? "p0" : "masked");
            print ("************************************************\n",comm);

            print (" -> creating tgt grid ...\n",comm);
            auto tgt_grid = src_grid->clone("tgt",true);
            tgt_grid->reset_vertical_configuration(nlevs_tgt, AbstractGrid::VKind::Pressure);
            print (" -> creating tgt grid ...done!\n",comm);

            print (" -> creating src/tgt pressure fields ...\n",comm);
            auto pint_src   = create_pint(src_grid, src_1d, ptop_src, pbot_src);
            auto pmid_src   = create_pmid(pint_src);

            // Make ptop_tgt<ptop_src and psurf_tgt>psurf_src, so we do have extrapolation
            const Real ptop_tgt = 10;
            const Real pbot_tgt = 1020;
            auto pint_tgt   = create_pint(tgt_grid, tgt_1d, ptop_tgt, pbot_tgt);
            auto pmid_tgt   = create_pmid(pint_tgt);
            print (" -> creating src/tgt pressure fields ... done!\n",comm);

            print (" -> creating fields ... done!\n",comm);
            auto src_s2d   = create_field("s2d",  src_grid,false,true,false);
            auto src_v2d   = create_field("v2d",  src_grid,false,true,true);
            auto src_s3d_m = create_field("s3d_m",src_grid,false,false,false,LEV, 1);
            auto src_s3d_i = create_field("s3d_i",src_grid,false,false,false,ILEV,SCREAM_PACK_SIZE);
            auto src_v3d_m = create_field("v3d_m",src_grid,false,false,true ,LEV, 1);
            auto src_v3d_i = create_field("v3d_i",src_grid,false,false,true ,ILEV,SCREAM_PACK_SIZE);

            auto tgt_s2d   = create_field("s2d",  tgt_grid,false,true,false);
            auto tgt_v2d   = create_field("v2d",  tgt_grid,false,true,true);
            auto tgt_s3d_m = create_field("s3d_m",tgt_grid,false,false,false,LEVP, 1);
            auto tgt_s3d_i = create_field("s3d_i",tgt_grid,false,false,false,LEVP, SCREAM_PACK_SIZE);
            auto tgt_v3d_m = create_field("v3d_m",tgt_grid,false,false,true ,LEVP, 1);
            auto tgt_v3d_i = create_field("v3d_i",tgt_grid,false,false,true ,LEVP, SCREAM_PACK_SIZE);

            // For expected fields, also create the mask extra data
            auto expected_s2d   = create_field("s2d",  tgt_grid,true,true,false);
            auto expected_v2d   = create_field("v2d",  tgt_grid,true,true,true);
            auto expected_s3d_m = create_field("s3d_m",tgt_grid,true,false,false,LEVP, 1);
            auto expected_s3d_i = create_field("s3d_i",tgt_grid,true,false,false,LEVP, SCREAM_PACK_SIZE);
            auto expected_v3d_m = create_field("v3d_m",tgt_grid,true,false,true ,LEVP, 1);
            auto expected_v3d_i = create_field("v3d_i",tgt_grid,true,false,true ,LEVP, SCREAM_PACK_SIZE);
            print (" -> creating fields ... done!\n",comm);

            // -------------------------------------- //
            //     Register fields in the remapper    //
            // -------------------------------------- //

            print (" -> creating and initializing remapper ...\n",comm);
            auto remap = std::make_shared<VerticalRemapper>(src_grid,tgt_grid);
            remap->set_source_pressure (pmid_src);
            remap->set_source_pressure (pint_src);
            remap->set_target_pressure (pmid_tgt);
            remap->set_target_pressure (pint_tgt);
            remap->set_extrapolation_type(etype_top,Top);
            remap->set_extrapolation_type(etype_bot,Bot);

            remap->register_field(src_s2d,  tgt_s2d);
            remap->register_field(src_v2d,  tgt_v2d);
            remap->register_field(src_s3d_m,tgt_s3d_m);
            remap->register_field(src_s3d_i,tgt_s3d_i);
            remap->register_field(src_v3d_m,tgt_v3d_m);
            remap->register_field(src_v3d_i,tgt_v3d_i);
            remap->registration_ends();
            print (" -> creating and initializing remapper ... done!\n",comm);

            // -------------------------------------- //
            //       Generate data for src fields     //
            // -------------------------------------- //

            print (" -> generate fields data ...\n",comm);
            compute_field(src_s2d,  pmid_src);
            compute_field(src_v2d,  pmid_src);
            compute_field(src_s3d_m,pmid_src);
            compute_field(src_s3d_i,pint_src);
            compute_field(src_v3d_m,pmid_src);
            compute_field(src_v3d_i,pint_src);

            // Pre-compute what we expect the tgt fields to be
            compute_field(expected_s2d,  pmid_tgt);
            compute_field(expected_v2d,  pmid_tgt);
            compute_field(expected_s3d_m,pmid_tgt);
            compute_field(expected_s3d_i,pint_tgt);
            compute_field(expected_v3d_m,pmid_tgt);
            compute_field(expected_v3d_i,pint_tgt);

            extrapolate(pmid_src,pmid_tgt,expected_s2d,  etype_top,etype_bot);
            extrapolate(pmid_src,pmid_tgt,expected_v2d,  etype_top,etype_bot);
            extrapolate(pmid_src,pmid_tgt,expected_s3d_m,etype_top,etype_bot);
            extrapolate(pint_src,pint_tgt,expected_s3d_i,etype_top,etype_bot);
            extrapolate(pmid_src,pmid_tgt,expected_v3d_m,etype_top,etype_bot);
            extrapolate(pint_src,pint_tgt,expected_v3d_i,etype_top,etype_bot);
            print (" -> generate fields data ... done!\n",comm);

            // -------------------------------------- //
            //              Perform remap             //
            // -------------------------------------- //

            // No bwd remap
            REQUIRE_THROWS(remap->remap_bwd());

            print (" -> run remap ...\n",comm);
            remap->remap_fwd();
            print (" -> run remap ... done!\n",comm);

            // -------------------------------------- //
            //          Check remapped fields         //
            // -------------------------------------- //

            using namespace Catch::Matchers;
            Real tol = 10*std::numeric_limits<Real>::epsilon();

            auto check = [&](const Field& computed, Field& expected) {
              auto rel_diff = computed.clone("rel_diff", CloneFlags::CopyData);
              rel_diff.update(expected,1,-1);
              // By construction, expected fields are always > 0, so we can divide
              rel_diff.scale_inv(expected);

              // Now set rel_diff=0 where mask==0, so we don't pollute the norm with masked-out entries
              auto mask = expected.get_valid_mask();
              rel_diff.deep_copy(0,mask,true);
              using Catch::Matchers::WithinAbs;
              REQUIRE_THAT (inf_norm(rel_diff).as<Real>(), WithinAbs(0,tol));
            };

            print (" -> check tgt fields ...\n",comm);
            check(tgt_s2d,expected_s2d);
            check(tgt_v2d,expected_v2d);
            check(tgt_s3d_m,expected_s3d_m);
            check(tgt_v3d_m,expected_v3d_m);
            check(tgt_s3d_i,expected_s3d_i);
            check(tgt_v3d_i,expected_v3d_i);
            print (" -> check tgt fields ... done!\n",comm);
          }
        }
      }
    }
  }

  // Clean up scorpio stuff
  scorpio::finalize_subsystem();

  print ("Testing vertical remapper ... done!\n",comm);
}

TEST_CASE ("vertical_remapper_conservative") {
  using namespace ShortFieldTagsNames;

  // Fields flagged as VerticalRemapper::Extensive hold a layer-integrated
  // quantity (e.g. a layer optical depth), so they must be remapped in a way
  // that preserves their column sum, rather than interpolated point-wise.

  ekat::Comm comm(MPI_COMM_WORLD);

  print ("Testing conservative vertical remapper ...\n",comm);

  constexpr int nldofs = 2;
  const int nlevs_src = 2*SCREAM_PACK_SIZE + 2;

  auto src_grid = build_grid(comm, nldofs, nlevs_src);

  // Build a p_int profile spanning [ptop,pbot], with layers that get thinner
  // toward the surface (as they do in the model), and the matching p_mid.
  auto make_pint = [&](const std::shared_ptr<AbstractGrid>& grid,
                       const Real ptop, const Real pbot) {
    const int nlevs = grid->get_num_vertical_levels();
    FieldIdentifier fid("p_int",grid->get_3d_scalar_layout(ILEV),ekat::units::Pa,grid->name());
    Field p (fid);
    p.get_header().get_alloc_properties().request_allocation(SCREAM_PACK_SIZE);
    p.allocate_view();
    auto v = p.get_view<Real**,Host>();
    for (int i=0; i<nldofs; ++i) {
      for (int k=0; k<=nlevs; ++k) {
        const Real s = static_cast<Real>(k)/nlevs;
        v(i,k) = ptop + (pbot-ptop)*std::pow(s,Real(1.6));
      }
    }
    p.sync_to_dev();
    return p;
  };
  auto make_pmid = [&](const Field& pint) {
    const auto& fid_int = pint.get_header().get_identifier();
    const int nlevs = fid_int.get_layout().dims().back()-1;
    auto grid_name = fid_int.get_grid_name();
    auto layout = fid_int.get_layout();
    layout.strip_dim(ILEV);
    layout.append_dim(LEV,nlevs);
    FieldIdentifier fid("p_mid",layout,ekat::units::Pa,grid_name);
    Field p (fid);
    p.get_header().get_alloc_properties().request_allocation(SCREAM_PACK_SIZE);
    p.allocate_view();
    auto vi = pint.get_view<const Real**,Host>();
    auto vm = p.get_view<Real**,Host>();
    for (int i=0; i<nldofs; ++i) {
      for (int k=0; k<nlevs; ++k) {
        vm(i,k) = 0.5*(vi(i,k)+vi(i,k+1));
      }
    }
    p.sync_to_dev();
    return p;
  };

  // Column sum of a 3d field, per (col[,comp])
  auto col_sums = [&](const Field& f) {
    f.sync_to_host();
    const auto& l = f.get_header().get_identifier().get_layout();
    const int nlevs = l.dims().back();
    std::vector<Real> sums;
    if (f.rank()==2) {
      auto v = f.get_view<const Real**,Host>();
      for (int i=0; i<nldofs; ++i) {
        Real s = 0;
        for (int k=0; k<nlevs; ++k) s += v(i,k);
        sums.push_back(s);
      }
    } else {
      auto v = f.get_view<const Real***,Host>();
      for (int i=0; i<nldofs; ++i) {
        for (int j=0; j<vec_dim; ++j) {
          Real s = 0;
          for (int k=0; k<nlevs; ++k) s += v(i,j,k);
          sums.push_back(s);
        }
      }
    }
    return sums;
  };

  // Fill a src field with rho(col,comp)*dp(k), i.e. a density that is constant
  // in pressure. A conservative remap must then return rho*dp_tgt exactly.
  auto fill_const_density = [&](const Field& f, const Field& pint) {
    const auto& l = f.get_header().get_identifier().get_layout();
    const int nlevs = l.dims().back();
    auto vi = pint.get_view<const Real**,Host>();
    if (f.rank()==2) {
      auto v = f.get_view<Real**,Host>();
      for (int i=0; i<nldofs; ++i) {
        for (int k=0; k<nlevs; ++k) v(i,k) = (i+1)*(vi(i,k+1)-vi(i,k));
      }
    } else {
      auto v = f.get_view<Real***,Host>();
      for (int i=0; i<nldofs; ++i) {
        for (int j=0; j<vec_dim; ++j) {
          for (int k=0; k<nlevs; ++k) v(i,j,k) = (i+1)*(j+2)*(vi(i,k+1)-vi(i,k));
        }
      }
    }
    f.sync_to_dev();
  };

  // Fill a src field with an arbitrary positive profile
  auto fill_bumpy = [&](const Field& f, const Field& pmid) {
    const auto& l = f.get_header().get_identifier().get_layout();
    const int nlevs = l.dims().back();
    auto vm = pmid.get_view<const Real**,Host>();
    auto val = [&](int i, int j, int k) {
      return (i+1)*(j+1)*(1 + std::sin(0.37*k) * 0.9) * std::exp(-vm(i,k)/50000.0) + 1e-3;
    };
    if (f.rank()==2) {
      auto v = f.get_view<Real**,Host>();
      for (int i=0; i<nldofs; ++i)
        for (int k=0; k<nlevs; ++k) v(i,k) = val(i,0,k);
    } else {
      auto v = f.get_view<Real***,Host>();
      for (int i=0; i<nldofs; ++i)
        for (int j=0; j<vec_dim; ++j)
          for (int k=0; k<nlevs; ++k) v(i,j,k) = val(i,j,k);
    }
    f.sync_to_dev();
  };

  const Real tol = 1e4*std::numeric_limits<Real>::epsilon();

  // Sweep over: tgt finer/coarser than src, and tgt column narrower/wider/equal
  // to the src column (so that the src column has to be clipped or padded).
  for (int nlevs_tgt : {nlevs_src/2, nlevs_src, 2*nlevs_src}) {
    for (auto ptop_pbot : { std::make_pair(Real(50),Real(1000)),   // same column
                            std::make_pair(Real(80),Real( 950)),   // narrower tgt
                            std::make_pair(Real(20),Real(1080))})  // wider tgt
    {
      const Real ptop_tgt = ptop_pbot.first;
      const Real pbot_tgt = ptop_pbot.second;
      const bool same_column = ptop_tgt==Real(50) and pbot_tgt==Real(1000);

      print ("************************************************\n",comm);
      print ("      nlevs src/tgt: %d / %d\n",comm,nlevs_src,nlevs_tgt);
      print ("      tgt column   : [%.0f, %.0f]\n",comm,ptop_tgt,pbot_tgt);
      print ("************************************************\n",comm);

      auto tgt_grid = src_grid->clone("tgt",true);
      tgt_grid->reset_vertical_configuration(nlevs_tgt, AbstractGrid::VKind::Model);

      auto pint_src = make_pint(src_grid,50,1000);
      auto pmid_src = make_pmid(pint_src);
      auto pint_tgt = make_pint(tgt_grid,ptop_tgt,pbot_tgt);
      auto pmid_tgt = make_pmid(pint_tgt);

      auto src_s3d = create_field("s3d",src_grid,false,false,false,LEV,SCREAM_PACK_SIZE);
      auto src_v3d = create_field("v3d",src_grid,false,false,true ,LEV,SCREAM_PACK_SIZE);
      auto tgt_s3d = create_field("s3d",tgt_grid,false,false,false,LEV,SCREAM_PACK_SIZE);
      auto tgt_v3d = create_field("v3d",tgt_grid,false,false,true ,LEV,SCREAM_PACK_SIZE);

      auto remap = std::make_shared<VerticalRemapper>(src_grid,tgt_grid);
      remap->set_source_pressure (pmid_src);
      remap->set_source_pressure (pint_src);
      remap->set_target_pressure (pmid_tgt);
      remap->set_target_pressure (pint_tgt);

      remap->register_field(src_s3d,tgt_s3d);
      remap->register_field(src_v3d,tgt_v3d);
      remap->set_remap_kind("s3d",VerticalRemapper::Extensive);
      remap->set_remap_kind("v3d",VerticalRemapper::Extensive);
      remap->registration_ends();

      // ---- 1. a density that is constant in pressure is reproduced exactly ---- //
      // NOTE: this only holds when src and tgt span the same column; otherwise
      //       the clipped part of the src column piles up in the edge layers.
      if (same_column) {
        print (" -> checking constant density is reproduced exactly ...\n",comm);
        fill_const_density(src_s3d,pint_src);
        fill_const_density(src_v3d,pint_src);
        remap->remap_fwd();

        auto expected_s3d = create_field("s3d",tgt_grid,false,false,false,LEV,SCREAM_PACK_SIZE);
        auto expected_v3d = create_field("v3d",tgt_grid,false,false,true ,LEV,SCREAM_PACK_SIZE);
        fill_const_density(expected_s3d,pint_tgt);
        fill_const_density(expected_v3d,pint_tgt);

        for (auto p : {std::make_pair(tgt_s3d,expected_s3d), std::make_pair(tgt_v3d,expected_v3d)}) {
          auto rel_diff = p.first.clone("rel_diff", CloneFlags::CopyData);
          rel_diff.update(p.second,1,-1);
          rel_diff.scale_inv(p.second);
          REQUIRE_THAT (inf_norm(rel_diff).as<Real>(), Catch::Matchers::WithinAbs(0,tol));
        }
        print (" -> checking constant density is reproduced exactly ... done!\n",comm);
      }

      // ---- 2. the column sum is preserved for an arbitrary profile ---- //
      print (" -> checking column sums are preserved ...\n",comm);
      fill_bumpy(src_s3d,pmid_src);
      fill_bumpy(src_v3d,pmid_src);
      remap->remap_fwd();

      for (auto p : {std::make_pair(src_s3d,tgt_s3d), std::make_pair(src_v3d,tgt_v3d)}) {
        auto s_src = col_sums(p.first);
        auto s_tgt = col_sums(p.second);
        REQUIRE (s_src.size()==s_tgt.size());
        for (size_t n=0; n<s_src.size(); ++n) {
          const Real rel = std::abs(s_tgt[n]-s_src[n])/std::abs(s_src[n]);
          REQUIRE (rel < tol);
        }
      }
      print (" -> checking column sums are preserved ... done!\n",comm);

      // ---- 3. a non-negative src gives a non-negative tgt ---- //
      print (" -> checking non-negativity ...\n",comm);
      tgt_s3d.sync_to_host();
      {
        auto v = tgt_s3d.get_view<const Real**,Host>();
        for (int i=0; i<nldofs; ++i) {
          for (int k=0; k<nlevs_tgt; ++k) {
            REQUIRE (v(i,k)>=0);
          }
        }
      }
      print (" -> checking non-negativity ... done!\n",comm);
    }
  }

  // ---- 4. interface fields cannot be flagged as extensive ---- //
  {
    auto tgt_grid = src_grid->clone("tgt",true);
    tgt_grid->reset_vertical_configuration(nlevs_src/2, AbstractGrid::VKind::Model);
    auto pint_src = make_pint(src_grid,50,1000);
    auto pmid_src = make_pmid(pint_src);
    auto pint_tgt = make_pint(tgt_grid,50,1000);
    auto pmid_tgt = make_pmid(pint_tgt);

    auto src_i = create_field("s3d_i",src_grid,false,false,false,ILEV,SCREAM_PACK_SIZE);
    auto tgt_i = create_field("s3d_i",tgt_grid,false,false,false,ILEV,SCREAM_PACK_SIZE);

    auto remap = std::make_shared<VerticalRemapper>(src_grid,tgt_grid);
    remap->set_source_pressure (pmid_src);
    remap->set_source_pressure (pint_src);
    remap->set_target_pressure (pmid_tgt);
    remap->set_target_pressure (pint_tgt);
    remap->register_field(src_i,tgt_i);
    remap->set_remap_kind("s3d_i",VerticalRemapper::Extensive);
    REQUIRE_THROWS (remap->registration_ends());
  }

  print ("Testing conservative vertical remapper ... done!\n",comm);
}

} // namespace scream
