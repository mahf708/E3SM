# NOTE: tests inside this if statement won't be built in a baselines-only build
if (NOT SCREAM_ONLY_GENERATE_BASELINES)
  include(ScreamUtils)

  ## Lexer/parser unit tests for the vendored e3sm_diags_parser package.
  ## These exercise the standalone package only: no EAMxx types involved.
  CreateUnitTest(edp_parser
    SOURCES test_lexer.cpp test_parser.cpp
    LIBS e3sm_diags_parser
    LABELS io diagnostics
  )
endif()
