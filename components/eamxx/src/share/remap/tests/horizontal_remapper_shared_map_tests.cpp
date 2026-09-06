#include <catch2/catch.hpp>

#include "share/remap/horizontal_remapper.hpp"
#include "share/grid/point_grid.hpp"
#include "share/scorpio_interface/eamxx_scorpio_interface.hpp"

#include <numeric>

namespace scream {

namespace {

void root_print (const std::string& msg, const ekat::Comm& comm) {
  if (comm.am_i_root()) {
    printf("%s",msg.c_str());
  }
}

// A coarsening map: every tgt dof averages two consecutive src dofs.
void create_remap_file (const std::string& filename, const int ngdofs_tgt)
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

std::shared_ptr<AbstractGrid>
build_src_grid (const ekat::Comm& comm, const int ngdofs)
{
  using gid_type = AbstractGrid::gid_type;
  const int nlevs = 20;

  int nldofs   = ngdofs / comm.size();
  int remainder = ngdofs % comm.size();
  int offset   = nldofs * comm.rank() + std::min(comm.rank(),remainder);
  if (comm.rank()<remainder) {
    ++nldofs;
  }

  auto grid = std::make_shared<PointGrid>("src",nldofs,nlevs,comm);
  auto dofs_h = grid->get_dofs_gids().get_view<gid_type*,Host>();
  std::iota(dofs_h.data(),dofs_h.data()+nldofs,offset+1);
  grid->get_dofs_gids().sync_to_dev();

  return grid;
}

// Mimic VerticalRemapper::create_tgt_grid: a shallow clone under a new name,
// with a different number of levels. Same GIDs, different object.
std::shared_ptr<AbstractGrid>
clone_as_vremap_tgt (const std::shared_ptr<AbstractGrid>& src_grid)
{
  auto g = src_grid->clone(src_grid->name()+"_vremap_tgt",true);
  g->reset_vertical_configuration(10, AbstractGrid::VKind::Pressure);
  return g;
}

} // anonymous namespace

// Finding 24. HorizRemapperDataRepo caches HorizRemapperData by map FILE and
// deliberately accepts a grid that is GID-identical but not the same object.
// HorizontalRemapper's single-grid constructor then has to decide which end of
// the map its grid sits on. Deciding that by pointer identity is wrong: a
// second remapper sharing the map file receives the first's cached data, and if
// its grid is a vertical-remap target clone the identity test fails and the
// remapper is built BACKWARDS -- silently, onto a same-size target.
//
// The repo holds weak_ptrs, so the bug needs both remappers ALIVE AT ONCE.
// That is why io_remap_test, which finalizes each output stream before creating
// the next, never caught it, and why E5b -- five concurrent output streams --
// did.
//
// This test asserts the TARGET COLUMN COUNT. Asserting only that a remapper was
// constructed, or that output appeared, passes against the bug.
TEST_CASE("horiz_remap_shared_map_file")
{
  ekat::Comm comm(MPI_COMM_WORLD);

  root_print ("\n +-------------------------------------------+\n",comm);
  root_print (" |   Horiz remap: two remappers, one map     |\n",comm);
  root_print (" +-------------------------------------------+\n\n",comm);

  // Catch2 re-runs the whole TEST_CASE body once per SECTION, so this is
  // entered several times; and a SECTION that fails never reaches the
  // finalize below. Guard both ends, or the first genuine failure is followed
  // by a confusing "re-initialize pio subsystem" cascade that buries it.
  if (not scorpio::is_subsystem_inited()) {
    scorpio::init_subsystem(comm);
  }

  const int nldofs_tgt = 3;
  const int ngdofs_tgt = nldofs_tgt*comm.size();
  const int ngdofs_src = ngdofs_tgt+1;

  std::string filename = "hr_shared_map_tests." + std::to_string(comm.size()) + ".nc";
  create_remap_file(filename, ngdofs_tgt);

  auto src_grid = build_src_grid(comm, ngdofs_src);

  // Both remappers must map src->tgt, regardless of which was built first and
  // regardless of whether the grid handed in is the original or a clone.
  SECTION ("plain first, clone second") {
    auto remap_plain = std::make_shared<HorizontalRemapper>(src_grid,filename);
    REQUIRE (remap_plain->get_tgt_grid()->get_num_global_dofs()==ngdofs_tgt);

    // Kept alive on purpose: this is what keeps the repo entry from expiring.
    auto vclone = clone_as_vremap_tgt(src_grid);
    auto remap_clone = std::make_shared<HorizontalRemapper>(vclone,filename);

    REQUIRE (remap_clone->get_src_grid()->get_num_global_dofs()==ngdofs_src);
    REQUIRE (remap_clone->get_tgt_grid()->get_num_global_dofs()==ngdofs_tgt);
  }

  SECTION ("clone first, plain second") {
    auto vclone = clone_as_vremap_tgt(src_grid);
    auto remap_clone = std::make_shared<HorizontalRemapper>(vclone,filename);
    REQUIRE (remap_clone->get_tgt_grid()->get_num_global_dofs()==ngdofs_tgt);

    auto remap_plain = std::make_shared<HorizontalRemapper>(src_grid,filename);

    REQUIRE (remap_plain->get_src_grid()->get_num_global_dofs()==ngdofs_src);
    REQUIRE (remap_plain->get_tgt_grid()->get_num_global_dofs()==ngdofs_tgt);
  }

  // A grid that is neither end of the map must be rejected, not silently
  // treated as the target.
  SECTION ("grid matching neither end is an error") {
    auto keep_alive = std::make_shared<HorizontalRemapper>(src_grid,filename);
    auto bogus = build_src_grid(comm, ngdofs_src);
    auto bogus_dofs_h = bogus->get_dofs_gids().get_view<AbstractGrid::gid_type*,Host>();
    for (int i=0; i<bogus->get_num_local_dofs(); ++i) {
      bogus_dofs_h(i) += 1000;
    }
    bogus->get_dofs_gids().sync_to_dev();

    REQUIRE_THROWS (std::make_shared<HorizontalRemapper>(bogus,filename));
  }

  if (scorpio::is_subsystem_inited()) {
    scorpio::finalize_subsystem();
  }
}

} // namespace scream
