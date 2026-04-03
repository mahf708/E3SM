# Tests for the standalone kokkos-diag-utils package
include(ScreamUtils)

# Expression parser tests (pure C++, no EAMxx)
CreateUnitTest(expression_parser_test
  "expression_parser_test.cpp"
  LIBS kokkos_diag_utils
  LABELS diagnostics_core
)
