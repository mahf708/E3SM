# e3sm_diags_parser (vendored)

A standalone lexer + Pratt parser for the E3SM diagnostics DSL, producing a
`std::variant`-based AST. Vendored into EAMxx as a **dependency-free** package.

## Provenance

- Upstream: <https://github.com/peterdschwartz/e3sm_diags_parser>
- Commit:   `d680d50e7b36dcfce46dec4d6f6181443c231219` (2026-08-06)
- License:  MIT (see `LICENSE`)

Layout is kept identical to upstream (`include/edp/`, `src/`, `tests/`,
`tools/`) so that re-syncing is a file copy plus re-application of the local
patches listed below.

## Ground rules

This package links against **nothing**: no EKAT, no Kokkos, no scorpio, no
EAMxx headers. It uses only the C++20 standard library. Please keep it that
way — any code that bridges the parser AST to EAMxx types (fields, grids,
`ekat::ParameterList`, the diagnostic factory) belongs in `eamxx_io`, not here.

## Local patches vs. upstream

Each of these is a candidate for upstreaming, after which it can be dropped:

1. `src/ast_print.cpp` — `std::format("{:e}", ...)` replaced with `snprintf`.
   `<format>` needs libstdc++ 13+ / libc++ 17+, which is not available across
   all compilers EAMxx must build with. Output is byte-identical.
2. `src/lexer.cpp` — `read_position_` (an `int`) is cast to `std::size_t`
   before comparing against `input_.length()`, silencing `-Wsign-compare`.
3. `src/parser.cpp` — `parse_expression` threw a bare `std::string` on an
   unexpected prefix token, which no `catch (const std::exception&)` handler
   can catch (it would `std::terminate` inside EAMxx). Now throws
   `ParserError`.
4. `src/tokens.cpp` — added the missing `<stdexcept>` include for
   `std::invalid_argument`.
5. `src/precedences.cpp` — added an explicit `default:` to the `switch`,
   silencing `-Wswitch`. Behavior is unchanged (it already fell through to the
   trailing `return Precedence::Lowest`).
6. `tests/` — ported from Catch2 v3 headers to the Catch2 v2 single header
   that EAMxx vendors, and driven by `CreateUnitTest` rather than upstream's
   `FetchContent`. Added a regression test for patch 3.

Upstream's empty `tests/test_list_supported_functions.cpp` was not copied.

## Building / testing

Built as part of EAMxx: target `e3sm_diags_parser`, unit test `edp_parser`.

```bash
ctest -R edp_parser
```

The upstream CLI is also built (target `edp`) when `SCREAM_LIB_ONLY` is off:

```bash
./edp functions   # list the DSL functions the parser recognizes
```

## Status

The parser is **not yet wired into diagnostic creation**. `create_diagnostic`
in `../eamxx_io_utils.cpp` still uses the hand-rolled regexes. Replacing them
requires a translation layer from the `edp::ast::Expression` tree to
diagnostic names + `ekat::ParameterList` params, plus a decision on how (or
whether) to keep accepting the existing `FIELD_at_500hPa`-style names. That
work is intentionally not part of this vendoring commit.
