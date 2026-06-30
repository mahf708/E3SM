#include <catch2/catch.hpp>

#include "physics/stochastic_forcing/eamxx_sppt_process_interface.hpp"
#include "physics/stochastic_forcing/eamxx_skebs_process_interface.hpp"

#include <ekat_comm.hpp>
#include <ekat_parameter_list.hpp>

// Construction / linkage smoke test for the stochastic-forcing processes.
// Full functional + conservation + restart coverage is exercised by the
// spectral-pattern unit tests (share/stochastic/tests) and by the CIME
// single-process and ERS system tests.

TEST_CASE ("stochastic_forcing_construction")
{
  using namespace scream;
  ekat::Comm comm(MPI_COMM_WORLD);

  SECTION ("sppt_begin") {
    ekat::ParameterList pl("sppt_begin");
    SPPTBegin proc(comm, pl);
    REQUIRE(proc.name() == "sppt_begin");
    REQUIRE(proc.type() == AtmosphereProcessType::Physics);
  }

  SECTION ("sppt") {
    ekat::ParameterList pl("sppt");
    pl.set<std::vector<std::string>>("perturbed_fields", {"T_mid","horiz_winds","qv"});
    SPPT proc(comm, pl);
    REQUIRE(proc.name() == "sppt");
    REQUIRE(proc.type() == AtmosphereProcessType::Physics);
  }

  SECTION ("skebs") {
    ekat::ParameterList pl("skebs");
    pl.set<std::string>("dissipation_source", "smoothed_shear");
    SKEBS proc(comm, pl);
    REQUIRE(proc.name() == "skebs");
    REQUIRE(proc.type() == AtmosphereProcessType::Physics);
  }
}
