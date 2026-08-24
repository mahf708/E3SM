#include <catch2/catch.hpp>

#include "share/remap/horizontal_remapper.hpp"
#include "share/grid/point_grid.hpp"
#include "share/scorpio_interface/eamxx_scorpio_interface.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace scream {

/*
 * A map file whose dst_grid_rank is 2 says the tgt grid is STRUCTURED, not that it is
 * RECTILINEAR. Any regional grid on a projection (Lambert conformal, rotated pole, ...)
 * is rank 2 but curvilinear: every column carries its own lat AND its own lon, so
 * ncol != nlat*nlon. Setting up a (lat,lon) output layout for such a grid declares
 * output vars with ~ncol^2 entries.
 *
 * These tests pin down that setup_latlon_data:
 *   - still produces the (lat,lon) layout, with exact coords, for a real rectilinear grid
 *   - tolerates the two SCRIP dst_grid_dims orderings
 *   - falls back to 'ncol' for curvilinear grids, including ones that only wander a
 *     small fraction of a cell
 *   - falls back to 'ncol' when dst_grid_dims is absent
 * and that lat_idx/lon_idx are a bijection onto [0,nlat*nlon), since they are used
 * directly as IO decomposition offsets.
 */

enum class TgtKind {
  Rectilinear,
  Curvilinear,       // every col its own lat/lon
  NearlyRectilinear, // rectilinear to within a small fraction of a cell
};

void write_latlon_map_file (const std::string& filename,
                            const int ngdofs_src,
                            const int nlat, const int nlon,
                            const TgtKind kind,
                            const bool write_dims,
                            const bool swap_dims)
{
  const int n_b = nlat*nlon;
  const int nnz = n_b;   // one src col per tgt col, weight 1

  scorpio::register_file(filename, scorpio::FileMode::Write);

  scorpio::define_dim(filename, "n_a", ngdofs_src);
  scorpio::define_dim(filename, "n_b", n_b);
  scorpio::define_dim(filename, "n_s", nnz);
  scorpio::define_dim(filename, "dst_grid_rank", 2);

  scorpio::define_var(filename, "col", {"n_s"}, "int");
  scorpio::define_var(filename, "row", {"n_s"}, "int");
  scorpio::define_var(filename, "S",   {"n_s"}, "double");
  if (write_dims) {
    scorpio::define_var(filename, "dst_grid_dims", {"dst_grid_rank"}, "int");
  }
  scorpio::define_var(filename, "yc_b",   {"n_b"}, "double");
  scorpio::define_var(filename, "xc_b",   {"n_b"}, "double");
  scorpio::define_var(filename, "area_b", {"n_b"}, "double");

  scorpio::enddef(filename);

  std::vector<int> col(nnz), row(nnz);
  std::vector<double> S(nnz,1.0);
  for (int i=0; i<nnz; ++i) {
    row[i] = i+1;                    // map files are 1-based
    col[i] = (i % ngdofs_src) + 1;
  }

  // Reference rectilinear coords
  std::vector<double> lats(nlat), lons(nlon);
  for (int j=0; j<nlat; ++j) lats[j] = -90.0 + 180.0*(j+0.5)/nlat;
  for (int i=0; i<nlon; ++i) lons[i] = 360.0*i/nlon;
  const double dlat = 180.0/nlat;
  const double dlon = 360.0/nlon;

  std::vector<double> yc(n_b), xc(n_b), area(n_b,4*M_PI/n_b);
  for (int g=0; g<n_b; ++g) {
    const int j = g/nlon;
    const int i = g%nlon;
    switch (kind) {
      case TgtKind::Rectilinear:
        yc[g] = lats[j];
        xc[g] = lons[i];
        break;
      case TgtKind::Curvilinear:
        // Skewed, like a projected regional grid
        yc[g] = lats[j] + 0.37*dlat*i;
        xc[g] = lons[i] + 0.53*dlon*j;
        break;
      case TgtKind::NearlyRectilinear:
        // Wanders only a few percent of a cell: the kind of grid a bounding-box
        // heuristic would wrongly accept
        yc[g] = lats[j] + 0.03*dlat*((i%3)-1);
        xc[g] = lons[i] + 0.03*dlon*((j%3)-1);
        break;
    }
  }

  scorpio::write_var(filename,"row",row.data());
  scorpio::write_var(filename,"col",col.data());
  scorpio::write_var(filename,"S",  S.data());
  if (write_dims) {
    // SCRIP convention is (nx,ny) == (nlon,nlat); some tools emit the swap
    std::vector<int> dims = swap_dims ? std::vector<int>{nlat,nlon}
                                      : std::vector<int>{nlon,nlat};
    scorpio::write_var(filename,"dst_grid_dims",dims.data());
  }
  scorpio::write_var(filename,"yc_b",  yc.data());
  scorpio::write_var(filename,"xc_b",  xc.data());
  scorpio::write_var(filename,"area_b",area.data());

  scorpio::release_file(filename);
}

TEST_CASE ("horiz_remapper_latlon_setup") {
  using gid_type = AbstractGrid::gid_type;

  ekat::Comm comm(MPI_COMM_WORLD);
  scorpio::init_subsystem(comm);

  const int nlat = 4;
  const int nlon = 2*comm.size();
  const int n_b  = nlat*nlon;
  const int ngdofs_src = 4*comm.size();   // != n_b, as the 1-grid ctor requires
  REQUIRE (ngdofs_src != n_b);

  const int nlevs = 4;
  auto src_grid = create_point_grid("src",ngdofs_src,nlevs,comm,1);

  const std::string sfx = ".np" + std::to_string(comm.size()) + ".nc";

  auto tgt_of = [&](const std::string& f) {
    HorizontalRemapper r(src_grid,f);
    return r.get_tgt_grid();
  };

  SECTION ("rectilinear grid keeps the (lat,lon) layout") {
    auto f = "ll_rect" + sfx;
    write_latlon_map_file(f,ngdofs_src,nlat,nlon,TgtKind::Rectilinear,true,false);
    auto tgt = tgt_of(f);

    REQUIRE (tgt->has_geometry_data("lat_idx"));
    REQUIRE (tgt->has_geometry_data("lon_idx"));

    const auto& lat = tgt->get_geometry_data("lat");
    const auto& lon = tgt->get_geometry_data("lon");
    REQUIRE (lat.get_header().get_identifier().get_layout().size()==nlat);
    REQUIRE (lon.get_header().get_identifier().get_layout().size()==nlon);

    // Coordinate values must be exactly what the map file stored
    auto lat_h = lat.get_view<const Real*,Host>();
    auto lon_h = lon.get_view<const Real*,Host>();
    for (int j=0; j<nlat; ++j) {
      CHECK (lat_h(j)==static_cast<Real>(-90.0 + 180.0*(j+0.5)/nlat));
    }
    for (int i=0; i<nlon; ++i) {
      CHECK (lon_h(i)==static_cast<Real>(360.0*i/nlon));
    }

    // lat_idx/lon_idx are used directly as IO decomposition offsets, so they must be
    // a bijection onto [0,n_b): offset(g) == g, with no two cols colliding.
    auto li_h = tgt->get_geometry_data("lat_idx").get_view<const int*,Host>();
    auto oi_h = tgt->get_geometry_data("lon_idx").get_view<const int*,Host>();
    auto gids_h = tgt->get_dofs_gids().get_view<const gid_type*,Host>();
    const auto gid_base = tgt->get_global_min_partitioned_dim_gid();
    for (int i=0; i<tgt->get_num_local_dofs(); ++i) {
      CHECK (li_h(i)*nlon + oi_h(i) == gids_h(i)-gid_base);
    }
  }

  SECTION ("swapped dst_grid_dims is still recognized") {
    auto f = "ll_swap" + sfx;
    write_latlon_map_file(f,ngdofs_src,nlat,nlon,TgtKind::Rectilinear,true,true);
    auto tgt = tgt_of(f);
    REQUIRE (tgt->has_geometry_data("lat_idx"));
    CHECK (tgt->get_geometry_data("lat").get_header().get_identifier().get_layout().size()==nlat);
    CHECK (tgt->get_geometry_data("lon").get_header().get_identifier().get_layout().size()==nlon);
  }

  SECTION ("curvilinear grid falls back to ncol") {
    auto f = "ll_curv" + sfx;
    write_latlon_map_file(f,ngdofs_src,nlat,nlon,TgtKind::Curvilinear,true,false);
    auto tgt = tgt_of(f);
    CHECK (not tgt->has_geometry_data("lat_idx"));
    CHECK (not tgt->has_geometry_data("lon_idx"));
    // lat/lon stay as per-column (COL) geo data
    CHECK (tgt->get_geometry_data("lat").get_header().get_identifier().get_layout().size()
             ==tgt->get_num_local_dofs());
  }

  SECTION ("nearly-rectilinear grid also falls back to ncol") {
    auto f = "ll_near" + sfx;
    write_latlon_map_file(f,ngdofs_src,nlat,nlon,TgtKind::NearlyRectilinear,true,false);
    auto tgt = tgt_of(f);
    CHECK (not tgt->has_geometry_data("lat_idx"));
    CHECK (not tgt->has_geometry_data("lon_idx"));
  }

  SECTION ("missing dst_grid_dims falls back to ncol") {
    auto f = "ll_nodims" + sfx;
    write_latlon_map_file(f,ngdofs_src,nlat,nlon,TgtKind::Rectilinear,false,false);
    auto tgt = tgt_of(f);
    CHECK (not tgt->has_geometry_data("lat_idx"));
    CHECK (not tgt->has_geometry_data("lon_idx"));
  }

  scorpio::finalize_subsystem();
}

} // namespace scream
