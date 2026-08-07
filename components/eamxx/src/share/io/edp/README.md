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

## Public API / behavior changes vs. upstream

Read this before writing anything against `edp::ast` or `edp::parser`:

- **`ast::ExpressionVariant` has a new alternative, `ast::SliceExpression`**
  (patch 19). *This breaks every exhaustive visitor over the AST.* A
  `std::visit` with one overload per alternative, or an overload set with no
  catch-all, will fail to compile until a `SliceExpression` case is added. Its
  three members `start`, `stop` and `step` are each an `ExprPtr` **that may be
  null**, meaning "component omitted" — not "zero". Null-check before
  dereferencing.
- **`Token` gained two members, `line` and `column`** (patch 17), both 1-based
  and pointing at the token's *first* character. They have default member
  initializers, so existing `{TokenTypes::X, "lit"}` aggregate initializations
  still compile (and claim position 1:1). `Token::type` also gained a default
  (`Illegal`) so a default-constructed `Token` is no longer indeterminate.
- **`ast::FloatLiteral::value` is now a `double`, not a `float`** (patch 13).
  Any future AST consumer — visitor, `std::visit` lambda, structured binding —
  sees a `double`. A visitor written against upstream that takes `float` by
  value still compiles (implicit narrowing conversion) but silently loses
  precision, so audit for it.
- **`Parser::parse()` now throws `ParserError` on trailing or illegal input**
  (patch 12). Upstream returned a partial expression. Callers that previously
  "succeeded" on malformed input will now see an exception. This is the point:
  `bc_a1` used to parse to `bc_a` with no diagnostic.
- **Keywords are lower-case only** (patch 9). `and`/`or`/`not` are operators;
  `AND`, `Not`, `OR` are ordinary identifiers. Upstream lower-cased the entire
  input, so any spelling worked — at the cost of mangling every field name.
- **Identifiers are case-preserving and may contain digits** (patches 9, 10).
  `T_mid`, `LiqWaterPath`, `bc_a1`, `O3` are each a single `Identifier` token
  with the literal spelled exactly as written.
- **The library never writes to `stdout`/`stderr`** (patch 12). Every failure
  is reported through `ParserError`. (`tools/edp.cpp` is a CLI and still
  prints.)
- **`ParserError::what()` is now multi-line per error** (patch 17): each entry
  is `line L, column C: <message>` followed by the offending source line and a
  caret. Anything that string-matches on the message text needs to cope with
  the position prefix; `find()` on the message body still works.
- **`Precedence::Bounds` moved** (patch 19), from between `Prefix` and `Call`
  to between `Lowest` and `Equal`. The enumerator values of `Equal` through
  `Prefix` therefore shifted by one. Nothing serializes these, but a caller
  that hard-coded an integer would be wrong.
- **`a = b = c` is now a parse error** (patch 18).

## Local patches vs. upstream

Each of these is a candidate for upstreaming, after which it can be dropped:

1. `src/ast_print.cpp` — `std::format("{:e}", ...)` replaced with `snprintf`.
   `<format>` needs libstdc++ 13+ / libc++ 17+, which is not available across
   all compilers EAMxx must build with. (The `"%e"` format itself was later
   replaced; see patch 13.)
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

Patches 7-16 are correctness fixes found while preparing the parser to replace
the hand-rolled regexes in `create_diagnostic`. Every one of them is reproduced
by a test in `tests/`.

7. `src/parser.cpp` — **`parse_prefix_expression` recursed forever.** Upstream
   captured the operator but never advanced past it:

   ```cpp
   auto op = cur_token_.type;
   auto right_expr = parse_expression(Precedence::Prefix);  // cur_token_ is
                                                            // still the op
   ```

   `parse_expression` then looked the *same* token up in `prefix_parse_fns_`
   and called `parse_prefix_expression` again, unboundedly. Symptom: parsing
   `-T` or `not x` overflowed the stack and segfaulted (SIGSEGV, no message).
   Fix: `next_token();` between the two lines. `-T` now yields `(-T)` and
   `not x` yields `(!x)`.

8. `src/parser.cpp` — **parenthesized grouping did not exist.**
   `Parser::parse_grouped_expression` was defined and declared but never
   inserted into `prefix_parse_fns_`, i.e. dead code. Symptom: `(T + q) * 2`
   threw `Unexpected Prefix Token {Type: LeftParen, Literal: (}`; *no* input
   with a leading `(` could parse, so operator precedence could not be
   overridden at all. Fix: register
   `{TokenTypes::LeftParen, &Parser::parse_grouped_expression}` in the prefix
   map. `LeftParen` is deliberately in both maps — in prefix position it means
   grouping, in infix position (already registered) it means a function call —
   which is the normal arrangement for a Pratt parser. `(T + q) * 2` now yields
   `((T+q)*2)`, distinct from `T + q * 2` -> `(T+(q*2))`.

9. `src/lexer.cpp`, `include/edp/tokens.hpp` — **the lexer destroyed the case
   of the input.** The constructor ran
   `std::transform(input_.begin(), input_.end(), input_.begin(), tolower)`.
   Symptom: EAMxx field names are case sensitive, so `T_mid` parsed to `t_mid`,
   `LiqWaterPath` to `liqwaterpath`, `AeroComCldTop` to `aerocomcldtop` — none
   of which name a real field. Fix: drop the `std::transform` (and the now
   unused `<algorithm>` include). Keyword recognition in `identifier_lookup`
   needs no change: the `keywords` map keys are already lower case, and Python
   keywords are lower-case only, so an exact match is the faithful behavior.
   Behavioral change: `AND`/`Not` used to lex as operators and now lex as
   identifiers. Consequence handled at the same time: `check_precision` only
   recognized a lower-case `'e'` exponent marker, which used to work for
   `1E-5` *only because* the input had been lower-cased; `'E'` is now accepted
   explicitly so `1E-5` still lexes as a Float.

10. `src/lexer.cpp` — **identifiers could not contain digits.**
    `is_valid_identifier` was `isalpha(ch) || ch == '_'` and was used both to
    dispatch in `next_token` and to scan in `read_identifier`. Symptom: aerosol
    and chemistry field names shattered into several tokens — `bc_a1` lexed as
    `Identifier(bc_a)` + `Integer(1)`, `so4_a2` as `so`, `4`, `_a`, `2`, and
    `O3` as `o`, `3`. Combined with patch 12's silent truncation, `bc_a1`
    parsed to just `bc_a`. Fix: split the predicate into `is_identifier_start`
    (letter or `_`) and `is_identifier_char` (alphanumeric or `_`), matching
    Python's rule that an identifier may contain but not begin with a digit.
    `read_identifier` consumes one start character then any number of
    continuation characters, and `next_token` dispatches on
    `is_identifier_start`. Numbers are unaffected: a digit-led run is still
    lexed by `read_number`, so `500` is an `Integer`, `1e-5` a `Float`, and
    `500hPa` is `Integer(500)` + `Identifier(hPa)`.

11. `src/lexer.cpp` — **`!` was not lexed at all.** `TokenTypes::NotEqual` was
    defined in the enum, given a precedence in `token_precedence`, registered
    in `infix_parse_fns_` and handled in `binary_op_to_string`, but
    `next_token` had no `case '!'`, so it could never be produced. Symptom:
    `qc != 0` produced an `Illegal` token and (per patch 12) silently parsed to
    `qc`. Fix: added `case '!'` producing `NotEqual` for `!=` and
    `TokenTypes::Bang` for a bare `!`, which is consistent with the `not`
    keyword already mapping to `Bang`.

12. `src/parser.cpp` — **two silent-failure bugs, the most damaging of the
    set.** (a) `Parser::next_token` reacted to an `Illegal` token with
    `std::cout << "Encountered Illegal Token: " << ...` and carried on — a
    library writing to stdout, and a diagnostic that is invisible in a batch
    job's log soup. (b) `Parser::parse()` called
    `parse_expression(Precedence::Lowest)` and returned without checking that
    the input had been consumed. Together, any input the lexer could not
    handle was truncated at the first bad token and the truncated prefix was
    returned as a valid-looking AST: `bc_a1` -> `bc_a`, `T @ x` -> `T`,
    `qc != 0` -> `qc`, `T_mid 500` -> `T_mid`. In production that means a typo
    in a user's YAML silently computes a *different diagnostic* than the one
    requested. Fix: (a) `next_token` calls the (previously declared but never
    defined) `Parser::add_error`, recording
    `"Illegal token in input: '<literal>'"`, and the `std::cout` and the
    `<iostream>` include are gone; (b) `parse()` requires `cur_token_` or
    `peek_token_` to be `EndofFile` and otherwise records
    `"Unexpected token after end of expression: <token>"`. Both surface
    through the existing `ParserError` and both messages carry the offending
    literal. Also folded in here: `parse_list_of_expressions` threw a bare
    `std::runtime_error` on an unterminated argument list, bypassing the
    `ParserError` channel every other failure uses; it now records the error
    and throws `ParserError`. And `parse_integer_literal` /
    `parse_float_literal` now check that `std::stoi`/`std::stod` consumed the
    entire literal and catch conversion failures, so a malformed or
    out-of-range numeric literal is a `ParserError` rather than a silently
    different number or an escaping `std::out_of_range`.

13. `include/edp/ast.hpp`, `src/parser.cpp`, `src/ast_print.cpp` — **float
    literals lost precision and printed unreadably.** `ast::FloatLiteral` held
    a `float`, `parse_float_literal` used `std::stof`, and `ToStringVisitor`
    printed `"%e"`. Symptom: `500.0` round-tripped as `5.000000e+02`, and
    `1e-5` as `1.000000e-05`; every literal was also quantized to ~7
    significant digits. Fix: the member is a `double`, parsing uses
    `std::stod`, and printing walks `%g`, `%.9g`, `%.16g`, `%.17g` and stops at
    the first whose text `strtod`s back to the exact same `double` — the
    shortest round-trip-stable form. `.0` is appended when the result would
    otherwise look like an integer, so the printed text re-lexes as a `Float`.
    `500.0` prints as `500.0`, `1e-5` as `1e-05`, `3.141592653589793` as
    itself. Neither `std::format` nor floating-point `std::to_chars` is used:
    `<format>` was already removed for portability (patch 1) and
    floating-point `to_chars` is missing from some of the same toolchains, so
    `snprintf` + `strtod` is the portable choice. **This changes a public
    header** — see the API section above.

14. `src/lexer.cpp` — **a float without an exponent lexed as an integer.**
    `next_token` classified a number as a `Float` only if its text contained
    `'e'`; everything else became an `Integer`, and `parse_integer_literal`
    then ran `std::stoi` on it. Symptom: `0.5` produced `Integer("0.5")` and
    parsed to `0`; `qc > 0.5` parsed to `(qc>0)`; `500.0` parsed to `500`. A
    threshold silently became a different threshold. Fix: classify as `Float`
    if the literal contains `'.'`, `'e'` or `'E'`. Related: `read_number`
    accepted any number of `'.'` characters, so `1.2.3` lexed as one token and
    `std::stod` would have truncated it to `1.2`; it now takes at most one
    `'.'`, and `1.2.3` becomes two number tokens that patch 12's end-of-input
    check rejects.

15. `src/lexer.cpp` — **three ways the lexer mishandled the end of its
    input.** (a) The `Illegal` branch of `next_token` returned without calling
    `read_char()`, so the lexer never advanced past the offending character and
    any `while (tok.type != EndofFile)` loop over `next_token` spun forever on
    input like `T @ x`. The parser only calls `next_token` a bounded number of
    times, which is why this was not fatal there, but it makes the `Lexer`
    unusable on its own. (b)
    `read_to_delim` looped `while (peek_char() != ch) read_char();` with no
    end-of-input check, so an unterminated string literal — say the plausible
    typo `T.mean(dim='lev)` — looped forever incrementing `read_position_`.
    (c) `check_precision` consumed an `'e'` unconditionally and then read one
    further character before recursing into `read_number`, so the input `1e`
    left `position_` *past* the end of the input and `read_number`'s
    `input_.substr(start_pos, position_ - start_pos)` threw a bare
    `std::out_of_range` from deep inside `basic_string`. Fix: (a) consume the
    character before returning the `Illegal` token; (b) stop the scan at
    end-of-input, and have `next_token` return an `Illegal` token when the
    closing delimiter was never found, which patch 12 turns into a
    `ParserError`; (c) only treat `e`/`E` as an exponent marker when an
    optionally signed digit sequence actually follows, and consume exactly the
    exponent digits. `1e` now lexes as `Integer(1)` + `Identifier(e)`, which
    patch 12's end-of-input check rejects with a readable message; `1.5e-8` and
    `2e3 + 1` are unaffected.

16. `src/tokens.cpp` — **printing a valid AST could throw.**
    `binary_op_to_string` had no case for `And`, `Or`, `GreaterEqual` or
    `LessEq`, all four of which the parser registers as infix operators, and
    fell through to `throw std::invalid_argument`. Symptom: `qc >= 1` parsed
    fine but `ast::to_string` on the result threw
    `Invalid Binary OperatorGreaterEqual`; likewise for
    `T_mid.where(qc > 1e-5 and qc < 1)`. Fix: added the four cases. `and`/`or`
    are printed space-padded (`(a and b)`) so the output stays lexable, unlike
    the symbolic operators which are printed tight (`(a+b)`), matching the
    existing convention.

Patches 17-19 are the error-reporting and syntax additions made while
preparing the parser to replace the hand-rolled regexes in `create_diagnostic`.
Two of them change the public API; see the section above.

17. `include/edp/tokens.hpp`, `include/edp/lexer.hpp`, `src/lexer.cpp`,
    `src/tokens.cpp`, `include/edp/parser.hpp`, `src/parser.cpp` — **no error
    ever said *where* the problem was.** The `Lexer` tracked only a byte
    offset, `Token` carried no position, and so every `ParserError` message was
    of the form `Illegal token in input: '@'` with nothing to locate it.
    Symptom: for a user-authored YAML entry like
    `T_mid.weighted('dp').mean(dim='lev'` the message named a token the user
    then had to hunt for by eye. Fix, in three parts:

    - `Token` gained `int line` and `int column`, both 1-based, both with
      default member initializers so the `{TokenTypes::X, "lit"}` aggregate
      initializations used throughout the sources and tests still compile.
      **This changes a public header** — see the API section above.
    - `Lexer` gained `line_`/`column_`, updated inside `read_char()` (the only
      place that consumes a character), with `column_` resetting on `'\n'`.
      This has to live in `read_char()` rather than in the `case '\n'` of
      `next_token`, because `skip_whitespace` eats newlines before that case
      can ever be reached. `next_token()` now records the position *after*
      skipping whitespace and *before* scanning, and stamps the result — the
      scanning body moved to a new private `scan_token()` so that its dozen
      early returns do not each have to remember to do it. The recorded
      position is therefore where the token *starts*, not where the lexer
      ended up. `identifier_lookup` also had to be fixed: it returns entries of
      the `keywords` table, which are position-less literals, so it now copies
      the scanned identifier's position onto the keyword token (`x and y`
      reports `and` at column 3, not column 1).
    - `Parser` gained `error_at(tok, msg)` / `add_error_at(tok, msg)`, and
      every error site — the `Illegal` branch of `next_token`,
      `expect_peek_and_advance`, `parse_expression`'s unexpected-prefix throw,
      the two numeric-literal validators, `parse_list_of_expressions` and
      `parse()`'s end-of-input check — routes through them. The rendering is
      `line L, column C: <msg>` plus a compiler-style caret snippet:

      ```
      Parser errors:
        - line 1, column 9: Illegal token in input: '@'
            T_mid + @foo
                    ^
      ```

      The snippet needs the original text, so `Lexer` gained a
      `const std::string& input() const` accessor (the `Parser` already owns
      the `Lexer` by value). Multi-line input shows only the offending line,
      with the column relative to that line; tabs are rendered as one space so
      the caret stays aligned; an end-of-input token parks the caret one past
      the last character; and an input with no such line (an empty string, say)
      degrades to the `line L, column C:` prefix with no snippet. Nothing is
      printed anywhere — it all goes into the `ParserError` message.

    Folded in: `parse_expression` used to `throw ParserError({one_message})`,
    discarding everything already recorded in `errors_`. For `T_mid + @foo`
    that threw away the *interesting* error (`Illegal token in input: '@'`) and
    kept only its consequence (`Unexpected Prefix Token {Type: Illegal, ...}`).
    It now appends and throws `ParserError(errors_)`. Also: the
    `expect_peek_and_advance` message read `Expected XGot {...}` (no
    separator); it now reads `Expected X, got {...}`.

18. `include/edp/parser.hpp`, `src/parser.cpp` — **`a = b = c` parsed
    happily.** `Assign` is registered as an ordinary infix operator, because
    that is how a keyword argument (`dim='lev'`) is spelled, and nothing
    stopped it from chaining. Symptom: `a = b = c` produced a nested
    `InfixExpression`, `((a=b)=c)`, that is meaningless in this DSL and that a
    later translation layer would have to detect and reject. Fix: `Assign` is
    now bound to a dedicated `parse_assign_expression` instead of the generic
    `parse_infix_expression`, and that function rejects the nesting
    structurally, in the parse path, rather than by post-walking the tree. It
    checks *both* operands, because the two spellings land on opposite sides:
    `Assign` is left-associative under the Pratt loop, so `a = b = c` arrives
    with an `Assign` on the **left**, while parenthesizing (`a = (b = c)`) puts
    one on the **right**. Both record
    `Chained assignment is not allowed: '=' is non-associative`, positioned at
    the offending `=`. Testing what kind of node an operand is needs the
    variant, which is private to `ast::Expression`, so a small `NodeKind`
    visitor goes through the public `visit`. `dim='lev'`, `f(a=1, b=2)` and
    `T_mid.mean(dim='lev', skipna=1)` are unaffected. `Equal` (`==`) is a
    different operator and is deliberately left alone: `a == b == c` still
    parses.

19. `include/edp/ast.hpp`, `src/ast_print.cpp`, `include/edp/precedences.hpp`,
    `src/precedences.cpp`, `include/edp/parser.hpp`, `src/parser.cpp` —
    **`TokenTypes::Colon` had a precedence but no parse function**, so no slice
    syntax parsed. Symptom: `T_mid.isel(lev=0:10)` failed with
    `Unexpected token after end of expression: {Type: Colon, Literal: :}`;
    upstream's intent survived only as a commented-out two-member
    `BoundsExpression` in `ast.hpp`. Fix, in four parts:

    - **A new AST node, `ast::SliceExpression { start; stop; step; }`, added to
      `ExpressionVariant`.** Three members, not upstream's two, so that
      `a[::2]` is expressible. Each is an `ExprPtr` that may be **null**,
      meaning the component was omitted (Python's `a[:10]`, `a[1:]`, `a[::2]`)
      — null is *not* zero, and `stop == nullptr` is materially different from
      `stop == IntegerLiteral{0}`. **Adding a variant alternative is a public
      API change for every AST visitor**; see the API section above.
    - `Precedence::Bounds` moved from between `Prefix` and `Call` down to
      between `Lowest` and `Equal`. Where it was, `:` bound *tighter* than
      unary minus and than `+`, so `-1:2` would have parsed as `-(1:2)` and
      `1+2:3` as `1+(2:3)`. Python binds the slice colon looser than every
      arithmetic and comparison operator, which is what the new position gives:
      `-1:2` is `(-1):2` and `1+2:3` is `(1+2):3`. It still binds tighter than
      `Assign`, so `lev=0:10` is the keyword argument `lev` with a slice value
      rather than a slice of `(lev=0)`.
    - `Colon` is registered in **both** parse maps, the way `LeftParen`
      already is: infix (`1:10`) and prefix (`:10`, `::2`, `:`). Both funnel
      into one `parse_slice_tail`, which consumes `stop` and an optional
      `:step` itself so the whole thing becomes a *single* `SliceExpression` —
      leaving the colon as plain left-associative infix would have turned
      `1:2:3` into `slice(slice(1,2),3)`. A component is present iff the token
      after the colon can begin an expression, which is asked of the prefix
      table rather than by enumerating terminators; that makes `)`, `]`, `,`,
      end-of-input and an `Illegal` token all answer "omitted" for free. A
      fourth component (`1:2:3:4`) records `Too many ':' in slice` rather than
      being silently dropped. Only syntax is enforced here; whether a given
      slice makes *sense* is the translation layer's problem. `slice(1, 10)`
      keeps parsing as the ordinary function call it always was.
    - **The printing trap.** `ast::to_string` output is the canonical identity
      string for an expression and gets re-parsed, so whatever a slice prints
      must read back as the *same* tree. Printed bare in Python form
      (`1:10`, `:10`, `1:`, `::2`, `1:10:2`, `:`), a slice is ambiguous
      wherever a neighbouring operator could steal the colon back: `(1:2)*3`
      would have printed `(1:2*3)` and re-parsed as `1:(2*3)`, and `-(1:2)`
      would have printed `(-1:2)` and re-parsed as `(-1):2`. So the *containing*
      node parenthesizes a slice child: `ToStringVisitor` routes the operands of
      `PrefixExpression`, `InfixExpression` and `FuncExpression`, and the
      components of a nested `SliceExpression`, through an `operand_to_string`
      that wraps slices in parentheses. Already-delimited contexts — call
      arguments and array elements — keep the bare form. A standalone slice
      therefore still prints exactly the Python spelling, `T_mid.isel(lev=0:10)`
      prints as `(T_mid.isel((lev=(0:10))))`, and `canonical(canonical(s))`
      remains equal to `canonical(s)` for every form. Note that an explicitly
      empty trailing component normalizes: `1::` and `1:10:` print as `1:` and
      `1:10`, and `::` prints as `:`. Those are the same slices in Python, and
      the normalized text is a fixed point.

Upstream's empty `tests/test_list_supported_functions.cpp` was not copied.

## Building / testing

Built as part of EAMxx: target `e3sm_diags_parser`, unit test `edp_parser`.

```bash
ctest -R edp_parser
```

`tests/test_lexer.cpp` and `tests/test_parser.cpp` carry a regression test for
each numbered patch above, plus end-to-end checks on the expression shapes this
parser exists to handle (`T_mid.weighted('dp').mean(dim='lev')`,
`T_mid.isel(lev=-1)`, `T_mid.where(qc > 1e-5)`, `(A + B) / C`, `abs(X)`).
One of them, "to_string round-trips (canonical form is a fixed point)", asserts
`canonical(canonical(s)) == canonical(s)` over a corpus that includes every
slice form; keep it green, because the translation layer treats the canonical
string as an expression's identity.
Because the package has no dependencies, it can also be built and tested
outside EAMxx with a plain compiler invocation over `src/*.cpp`, `tests/*.cpp`
and a TU defining `CATCH_CONFIG_MAIN`.

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

What the parser *can* now do, which it could not as vendored: handle real
EAMxx field names (case-preserving, digits allowed), parse grouped and prefix
expressions, produce `!=`, parse Python-style colon slices, reject chained
assignment, point at the offending line and column of a bad input, and — most
importantly for a translation layer — fail loudly instead of silently returning
a truncated expression. A caller can rely on "either `parse()` returns an AST
for the whole input, or it throws `ParserError`", and on
`to_string(parse(s))` being a fixed point, so the canonical string can be used
as an expression's identity.

Known remaining rough edges, none of which are regressions and none of which
were fixed here:

- `TokenTypes::Newline`, `Concat`, `Semicolon`, `Percent` and `DoubleColon` are
  never produced by the lexer (`'\n'` is eaten by `skip_whitespace` before the
  `case '\n'` can be reached). A `::` is two `Colon` tokens, which is what the
  slice parser wants anyway.
- A keyword argument in a non-call position (`dim='lev'` on its own) still
  parses, as does a comparison chain (`a == b == c`, left-associatively).
  Only `=` was made non-associative; the translation layer, not the parser,
  decides where a keyword argument is meaningful.
- Slices are accepted anywhere an expression is, so syntactically valid but
  semantically empty things like `T_mid.:` or `T_mid:` parse. That is
  deliberate — the parser enforces syntax, the translation layer enforces
  meaning — but it means the translation layer must reject a `SliceExpression`
  in a position it cannot use, rather than assuming the parser did.
- The caret snippet renders the offending line in full, with no windowing. A
  pathologically long single-line input produces a pathologically long error
  message. YAML diagnostic entries are short, so this has not been worth
  fixing.
