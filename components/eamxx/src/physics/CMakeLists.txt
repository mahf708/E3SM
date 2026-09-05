add_library(eamxx_physics INTERFACE)

# TEMPORARY, for the dexpr diagnostics campaign only. Do not upstream.
#
# ZM and GW are not used by any configuration this campaign runs, but their
# generated test-data translation units (zm_test_data.cpp, gw_test_data.cpp)
# take minutes EACH under nvcc and dominate a GPU build. Both parametrizations
# are registered behind EAMXX_HAS_ZM / EAMXX_HAS_GW, and those defines come
# from the targets themselves, so dropping the subdirectories disables
# registration cleanly -- register_physics.hpp was written for exactly this.
#
# Set EAMXX_CAMPAIGN_SKIP_ZM_GW=OFF to build them again.
set(EAMXX_CAMPAIGN_SKIP_ZM_GW ON CACHE BOOL "Skip ZM and GW (dexpr campaign)")

# Add all physics parametrizations subfolders
add_subdirectory(p3)
if (SCREAM_DOUBLE_PRECISION)
  add_subdirectory(rrtmgp)
  add_subdirectory(cosp)
  add_subdirectory(tms)
  add_subdirectory(iop_forcing)
  if (NOT EAMXX_CAMPAIGN_SKIP_ZM_GW)
    add_subdirectory(zm)
  endif()
else()
  message(STATUS "WARNING: RRTMGP, COSP, TMS, ZM, and IOPForcing only supported for double precision builds; skipping")
endif()
add_subdirectory(shoc)
add_subdirectory(cld_fraction)
add_subdirectory(spa)
add_subdirectory(spc)
add_subdirectory(nudging)
if (SCREAM_ENABLE_MAM)
  add_subdirectory(mam)
endif()
if (NOT EAMXX_CAMPAIGN_SKIP_ZM_GW)
  add_subdirectory(gw)
endif()
add_subdirectory(specialized_tracers)
