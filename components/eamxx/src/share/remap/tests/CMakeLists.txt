# NOTE: tests inside this if statement won't be built in a baselines-only build
if (NOT SCREAM_ONLY_GENERATE_BASELINES)
  include(ScreamUtils)

  # Test iop remap
  CreateUnitTest(iop_remapper
    SOURCES iop_remapper_tests.cpp
    LIBS eamxx_remap
    MPI_RANKS 1 ${SCREAM_TEST_MAX_RANKS}
  )

  # Test coarsening remap
  CreateUnitTest(coarsening_remapper
    SOURCES coarsening_remapper_tests.cpp
    LIBS eamxx_remap
    MPI_RANKS 1 ${SCREAM_TEST_MAX_RANKS}
  )

  # Test refining remap
  CreateUnitTest(refining_remapper
    SOURCES refining_remapper_tests.cpp
    LIBS eamxx_remap
    MPI_RANKS 1 ${SCREAM_TEST_MAX_RANKS}
  )

  # Test that two horiz remappers sharing one map file both map src->tgt,
  # in either construction order (finding 24).
  CreateUnitTest(horizontal_remapper_shared_map
    SOURCES horizontal_remapper_shared_map_tests.cpp
    LIBS eamxx_remap
    MPI_RANKS 1 ${SCREAM_TEST_MAX_RANKS}
  )

  # Test vertical remap
  CreateUnitTest(vertical_remapper
    SOURCES vertical_remapper_tests.cpp
    LIBS eamxx_remap
    MPI_RANKS 1 ${SCREAM_TEST_MAX_RANKS}
  )
endif()
