##################################################
#   E3SM Diags Parser (vendored, standalone)     #
##################################################

# A self-contained lexer/parser for the E3SM diagnostics DSL. This library is
# deliberately dependency-free: it links against nothing (no EKAT, no Kokkos,
# no scorpio) and uses only the C++ standard library, so that it stays easy to
# keep in sync with its upstream repo. See README.md for provenance.
#
# Do NOT add EAMxx dependencies here. Code that needs to bridge the parser AST
# to EAMxx types belongs in eamxx_io (or another EAMxx lib), not in here.

add_library(e3sm_diags_parser
  src/lexer.cpp
  src/ast_print.cpp
  src/tokens.cpp
  src/precedences.cpp
  src/parser.cpp
)

target_compile_features(e3sm_diags_parser PUBLIC cxx_std_20)

target_include_directories(e3sm_diags_parser PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (NOT SCREAM_LIB_ONLY)
  # Small CLI to introspect the DSL (e.g. `edp functions`). Not used by EAMxx.
  add_executable(edp tools/edp.cpp)
  target_link_libraries(edp PRIVATE e3sm_diags_parser)

  add_subdirectory(tests)
endif()
