if (NOT SCREAM_ONLY_GENERATE_BASELINES)
  include(ScreamUtils)

  CreateUnitTest(stochastic_forcing
    SOURCES stochastic_forcing_tests.cpp
    LIBS stochastic_forcing
    LABELS physics stochastic
  )

endif()
