if (NOT SCREAM_ONLY_GENERATE_BASELINES)
  include(ScreamUtils)

  # Pure-math validation of the spherical-harmonic evaluation (host-only).
  CreateUnitTest(associated_legendre
    SOURCES associated_legendre_tests.cpp
    LIBS eamxx_stochastic
    LABELS stochastic
  )

  # Spectral pattern generator: determinism, restart reproducibility, AR(1).
  CreateUnitTest(spectral_pattern
    SOURCES spectral_pattern_tests.cpp
    LIBS eamxx_stochastic
    MPI_RANKS 1 ${SCREAM_TEST_MAX_RANKS}
    LABELS stochastic
  )

endif()
