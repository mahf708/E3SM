include(${CMAKE_CURRENT_LIST_DIR}/common.cmake)
common_setup()

set(CMAKE_Fortran_FLAGS "-Wno-maybe-uninitialized -Wno-unused-dummy-argument -fallow-argument-mismatch"  CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "-fvisibility-inlines-hidden -fmessage-length=0 -Wno-use-after-free -Wno-unused-variable -Wno-maybe-uninitialized" CACHE STRING "" FORCE)

set(EKAT_MPI_NP_FLAG "-np" CACHE STRING "-np")

set(ENV{CCSM_CPRNC} "/usr/local/packages/bin/cprnc")
