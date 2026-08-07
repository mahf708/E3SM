# Diagnostics DSL: replacing the name regexes with a real parser

Status: **design / scope**. Nothing here is implemented yet beyond the vendored
parser itself (see `src/share/io/edp/README.md`).

## Goal

Replace the hand-rolled regexes in `create_diagnostic`
(`src/share/io/eamxx_io_utils.cpp`) with the vendored `e3sm_diags_parser`
(`edp`), and expose a diagnostics syntax that reads like Python/xarray rather
than like a filename. Existing names keep working through a quarantined
compatibility shim.

## 1. Why the regexes have to go

`create_diagnostic` is ~15 regexes tested in a fixed order, and the order is
load-bearing. `docs/user/diags/parsing_precedence.md` documents the resulting
rules, and they are exactly the rules a grammar should be making for us:

- `_over_dt` must be tested before binary ops, or `X_over_dt` becomes
  `BinaryOp(X, over, dt)`.
- Binary ops must be tested before `_prev`, or `X_minus_X_prev` becomes
  `FieldPrev(X_minus_X)`.
- The left operand is greedy, so `A_minus_B_over_C` silently means
  `(A - B) / C` and there is no way to write `A - (B / C)`.

There is no grouping, no precedence the user can override, and every new
diagnostic adds another ordering constraint against all the existing ones. The
combinatorics get worse with each diag, and mistakes are silent rather than
loud.

## 2. The constraint that shapes everything: names are the ABI

This is the single most important thing to understand before planning the work.

Diagnostics do **not** hold references to their child diagnostics. They hold
**strings**. `BinaryOp` stores `m_arg1_name` and pushes it into
`m_field_in_names`; `AtmosphereOutput::set_diagnostics` then loops, and for any
dependency name not present in the field manager it calls `create_diagnostic`
on that name and adds the result to the FM
(`src/share/io/scorpio_output.cpp`, the `while (not done)` loop).

Two consequences:

1. **A diag's output field name must equal the string its parent asked for.**
   Every composite diag reconstructs the legacy name by concatenation —
   `m_field_name + "_at_" + p_value + units` in `field_at_pressure_level.cpp`,
   `m_field_name + "_vert_" + m_contract_method` in `vert_contract.cpp`,
   `m_input_f + "_where_" + ...` in `conditional_sampling.cpp`, and so on
   across ~15 classes. The legacy naming convention is not confined to
   `create_diagnostic`; it is baked into the diags themselves.
2. **All diag params are `std::string`.** Confirmed across every
   `m_params.get<...>` call site in `share/diagnostics`. So an AST-to-params
   translation only ever has to produce strings.

This means the cheap version of this project is possible: keep the string
protocol, put the parser in front of it. The expensive version — diags that
consume structured params and no longer name themselves — is a separate,
later, much larger change. **Recommendation: do the cheap version first**, and
treat decoupling names from diags as a follow-on only if it earns its keep.

## 3. Parser readiness: defects found

I probed the vendored parser against real EAMxx field names and Python-shaped
input. It is a sound Pratt parser with a clean AST, but it is not yet usable
against this codebase. Everything below is verified by running it, not by
reading it.

| # | Defect | Repro | Effect |
|---|--------|-------|--------|
| P1 | `parse_prefix_expression` never advances past the operator, so it recurses on itself forever | `-T` | **Segfault** (stack overflow). Kills all unary minus and `not`. |
| P2 | `parse_grouped_expression` exists but is never registered in `prefix_parse_fns_` | `(T + q) * 2` | Parse error. **No parenthesized grouping at all** — the main thing we want over the regexes. |
| P3 | Lexer lowercases the entire input | `T_mid` → `t_mid`, `LiqWaterPath` → `liqwaterpath` | Every case-sensitive field and diag name is destroyed. |
| P4 | Identifiers are `isalpha \|\| '_'` — digits excluded | `bc_a1` → `bc_a`,`1`; `so4_a2` → `so`,`4`,`_a`,`2`; `O3` → `o`,`3` | All aerosol/chemistry field names break. |
| P5 | Lexer has no `'!'` case, so `TokenTypes::NotEqual` can never be produced | `qc != 0` | `!=` is an Illegal token despite being wired through precedences and printing. |
| P6 | Illegal tokens are only `std::cout`-printed, and `parse()` does not require EOF | `bc_a1` → `bc_a`; `T @ x` → `t`; `qc != 0` → `qc` | **Silent truncation to a different, valid-looking expression.** A typo in a YAML field name yields the wrong diagnostic with no error. This is the most dangerous one. |
| P7 | `FloatLiteral` is `float`, printed as `%e` | `500.0` → `5.000000e+02` | Lossy round-trip; ugly if float literals ever appear in canonical names. |

P1 and P2 are one-line fixes each (add `next_token()`; register `LeftParen` in
`prefix_parse_fns_`). I verified both fixes locally. P3–P7 are small but need
tests. All seven are upstreamable to `peterdschwartz/e3sm_diags_parser`.

## 4. Proposed syntax (Python / xarray)

The parser already supports method chaining, keyword arguments, arrays, string
literals, comparisons, and `and`/`or`/`not` — it is genuinely well suited to an
xarray-shaped API. Verified working today (post-P1/P2 fix):

```text
T_mid.weighted('dp').sum(dim='lev')   →  ((t_mid.weighted('dp')).sum((dim='lev')))
T_mid.isel(lev=-1)                    →  (t_mid.isel((lev=(-1))))
T_mid.sel(lev=[1,2,3])                →  (t_mid.sel((lev=[1, 2, 3])))
```

Proposed mapping from today's names:

| Today | Proposed | xarray precedent |
|-------|----------|------------------|
| `X_at_lev_10` | `X.isel(lev=10)` | `.isel` |
| `X_at_model_top` / `X_at_model_bot` | `X.isel(lev=0)` / `X.isel(lev=-1)` | Python negative indexing |
| `X_at_500hPa` | `X.interp(plev=500, units='hPa')` | `.interp` (it does interpolate) |
| `X_at_10m_above_surface` | `X.interp(z=10, reference='surface')` | `.interp` |
| `X_horiz_avg` | `X.mean(dim='col')` | `.mean(dim=)` |
| `X_vert_avg` / `X_vert_sum` | `X.mean(dim='lev')` / `X.sum(dim='lev')` | `.mean` / `.sum` |
| `X_vert_avg_dp_weighted` | `X.weighted('dp').mean(dim='lev')` | `.weighted(w).mean()` |
| `X_zonal_avg_20_bins` | `X.zonal_mean(bins=20)` | (`.groupby_bins('lat', 20).mean()` later) |
| `X_where_Y_gt_0` | `X.where(Y > 0)` | `.where` |
| `A_plus_B`, `A_over_B` | `A + B`, `A / B` | operators |
| `X_prev` | `X.shift(time=1)` | `.shift` |
| `X_over_dt` | `X / dt` | operators |
| `X_atm_backtend` | `X.tend()` | already in `supported_functions.hpp` |
| `X_pvert_derivative` | `X.differentiate('p')` | `.differentiate(coord)` |
| `X_histogram_0_1_2` | `X.histogram(bins=[0,1,2])` | xhistogram |
| `log_X`, `abs_X` (hbc2) | `log(X)`, `abs(X)` | numpy free functions |
| `X_below` / `X_above` (hbc2) | `X.below()` / `X.above()` | — |
| `LiqWaterPath`, `z_mid` | unchanged bare identifiers | — |

Grouping now works, so `A - (B / C)` is expressible and greedy-left-operand
folklore goes away.

**netCDF naming.** `T_mid.mean(dim='lev')` is not a legal netCDF variable name.
DSL expressions in `field_names` must therefore be aliased using the existing
`:=` mechanism, which already supports exactly this:

```yaml
field_names:
  - T_vavg := T_mid.weighted('dp').mean(dim='lev')
```

This should be enforced with a clear error, not discovered at write time.

## 5. Backward compatibility

Per the requirement that built-in aliases are an acceptable compat mechanism:

Resolution order in the new `create_diagnostic` becomes:

1. Parse the requested string as DSL.
2. If it parses to a **bare identifier** that is neither a model field nor a
   registered diag product, hand it to the legacy shim.
3. The shim returns a DSL string (`T_mid_at_500hPa` →
   `T_mid.interp(plev=500, units='hPa')`), which is re-parsed and built.
4. Anything else is an error, loudly.

Note this only works once P4 is fixed — `T_mid_at_500hPa` must lex as a single
identifier for step 2 to see it whole.

The shim lives in **one quarantined translation unit**
(`legacy_diag_names.cpp`) exposing a single
`std::optional<std::string> legacy_to_dsl(const std::string&)`. It may keep
using regex internally; the point is that it is a pure syntactic rewrite with
no param extraction, no ordering coupling to the diag factory, independently
testable, and deletable in one commit when the deprecation window closes. That
is a very different object from the current 15 interleaved regexes.

## 6. Diagnostics to bring forward from `mahf708/E3SM@hbc2`

`hbc2` is 27 commits and predates the `src/diagnostics` →
`src/share/diagnostics` move and the `AtmosphereDiagnostic` →
`AbstractDiagnostic` API change, so every port needs adaptation — none of these
are clean cherry-picks.

**Worth porting (real gaps):**

- `UnaryOpsDiag` — `log`, `exp`, `sqrt`, `abs`, `square`, `inverse`. Master has
  no unary ops at all. Fills `log(X)`/`abs(X)` in the table above. Note the
  masking for `log`/`sqrt` must be rewritten onto master's
  `create_valid_mask()` / `set_may_be_filled()` API.
- `BelowOrAboveInterface` — mid/interface level shifting; no equivalent on
  master.
- `vert_contract` extended to `min`, `max`, `var`, `std` (master has only
  `avg`/`sum`). Directly feeds `.min()`/`.max()`/`.std()`/`.var()`.
- `PBLEntrainmentBudget` (+ its 297-line test) — a science diag rather than a
  DSL gap, but it is gone from master and the branch title is "bring back
  pblediags from the dead".
- `isccp_meanptop` exposed from COSP.

**Do not port (already superseded on master):**

- `binary_ops` `times_rho_h2o` / `over_gravit` special-casing. Master's
  `BinaryOp` already resolves constants generically via
  `physics::Constants<Real>::dictionary()`, so `Rgas_times_T_mid` works and the
  hbc2 hack is obsolete.
- The `mask_field` / `mask_value` extra-data unification commits. Master has
  since moved to `get_valid_mask()` / `create_valid_mask()` /
  `set_may_be_filled()`, which is a newer mechanism.

## 7. Work plan (isolated commits)

Grouped into three PRs so each is separately reviewable and revertable.

**PR 1 — make the parser correct (no EAMxx behavior change).**

1. Fix P1: advance past the prefix operator. Add a `-T` / `not x` test.
2. Fix P2: register `LeftParen` as a prefix fn. Add a `(a+b)*c` test.
3. Fix P3: stop lowercasing; make only keyword lookup case-insensitive.
4. Fix P4: allow digits in identifiers (not as the first character).
5. Fix P5: lex `!` and `!=`.
6. Fix P6: make Illegal tokens an error, and require EOF in `parse()`. Add
   silent-truncation regression tests (`bc_a1`, `T @ x`).
7. Fix P7: `FloatLiteral` to `double`, round-trip-stable printing.
8. Open the corresponding upstream PR; update `edp/README.md` patch list.

**PR 2 — bring forward the hbc2 diags (still on legacy names).**

9. Port `UnaryOpsDiag` onto the current `AbstractDiagnostic` + valid-mask API.
10. Port `BelowOrAboveInterface`.
11. Extend `vert_contract` with `min`/`max`/`var`/`std`.
12. Port `PBLEntrainmentBudget` and its test.
13. Expose `isccp_meanptop`.

Each with its legacy-name regex entry, so PR 2 is independently useful even if
PR 3 slips.

**PR 3 — switch `create_diagnostic` to the parser.**

14. Add the AST → `(diag_name, ParameterList)` walker, with the canonical-name
    printer that renders sub-expressions back to legacy child names (needed by
    §2's string ABI). Unit-tested standalone.
15. Add `legacy_diag_names.cpp` shim + exhaustive table test asserting every
    name in `create_diag.cpp` and in every in-repo YAML maps to the same diag
    and params as today.
16. Swap `create_diagnostic` to parse-first/shim-second. Delete the 15 regexes.
17. Enforce the `:=` alias requirement for non-identifier DSL expressions, with
    a clear error message.
18. Docs: rewrite `parsing_precedence.md` around the grammar, add a DSL
    reference page, mark legacy names deprecated.

## 8. Integration design for `create_diagnostic`

### 8.1 The unlock: an optional `output_name` param

§2 said diags name themselves, so a parent's param string must equal the
child's self-chosen name. That is what would otherwise force the walker to
render sub-expressions back into legacy names.

There is a much cheaper way out. Add one protected helper to
`AbstractDiagnostic`:

```cpp
// Name of the diag output field. Defaults to the legacy auto-generated name;
// the DSL walker overrides it so that a diag's output name matches the string
// its parent asked for.
std::string output_name (const std::string& default_name) {
  return m_params.get<std::string>("output_name", default_name);
}
```

and change each diag's one naming line:

```cpp
-  auto diag_name = m_field_name + "_at_" + location;
+  auto diag_name = output_name(m_field_name + "_at_" + location);
```

That is ~15 mechanical one-line edits with **zero behavior change when
`output_name` is absent**. `m_params` is already protected and the two-argument
`ParameterList::get` is already used elsewhere (`vert_contract.cpp` does it for
`weighting_method`). This is emphatically *not* the big "decouple names from
diags" project I recommended deferring — that one changes call sites and
signatures. This is a defaulted opt-in.

With it, the DSL string itself becomes the canonical name and no
legacy-name rendering is needed anywhere.

### 8.2 Resolution flow

```text
create_diagnostic(name, grid):
  1. expr = edp::parser::Parser{edp::Lexer{name}}.parse()      # throws on garbage
  2. if expr is a bare Identifier:
        - if it is a registered diag product  -> build it directly (z_mid, LiqWaterPath)
        - else                                -> legacy_to_dsl(name), re-parse, but keep
                                                 output_name = name (the legacy string)
  3. spec = walk(expr, grid)                                   # -> (diag_name, params)
  4. params.set("output_name", canonical(expr))
  5. DiagnosticFactory::instance().create(spec.diag_name, comm, params, grid)
```

Step 4 is what closes the loop in `scorpio_output.cpp`. The walker sets a
child's `field_name` param to `canonical(child_expr)`; the output loop finds
that name missing from the FM and calls `create_diagnostic` on it; that call
parses the DSL string directly (no shim round-trip) and produces a diag whose
output field is named with the *same* canonical string. The FM lookup then
succeeds and the recursion terminates.

Keeping `output_name = name` in step 2 for legacy inputs means existing runs
keep byte-identical netCDF variable names. That is worth protecting.

One accepted wart: if a user requests both `f_minus_f_prev` (legacy) and the
equivalent DSL expression in the same stream, the FM holds two entries
computing the same thing under different names. Wasteful, not wrong.

### 8.3 Canonical form must be idempotent

`canonical(expr)` is the string identity of a diagnostic, so it must satisfy

```text
canonical(parse(s)) == canonical(parse(canonical(parse(s))))
```

or the recursion in 8.2 never converges. For v1, reuse the existing
`ast::to_string` (it is fully parenthesized and therefore unambiguous) and add
a **property test** asserting idempotency over a corpus of expressions. A
precedence-aware pretty printer that emits `T_mid.mean(dim='lev')` instead of
`(t_mid.mean((dim='lev')))` is a nice-to-have, not a blocker — these strings
are internal and get aliased before reaching netCDF.

### 8.4 What the AST actually looks like

Method chains do **not** produce a dedicated node. `T.mean(dim='lev')` parses
as `Infix(left=T, op=Dot, right=Func(function=mean, args=[...]))`, because
`LeftParen` has `Call` precedence, which binds tighter than `Dot`'s `Prefix`
precedence. Chains nest left-associatively:

```text
T.weighted('dp').sum(dim='lev')
  -> Infix(Infix(T, Dot, Func(weighted, ['dp'])), Dot, Func(sum, [dim='lev']))
```

Keyword arguments are `Infix(Identifier, Assign, <literal>)` inside the args
vector. So the walker needs one shared helper that splits an args vector into
positional and keyword arguments; everything else is a dispatch table on the
method name.

### 8.5 Dispatch table

| DSL | Diag | Params |
|-----|------|--------|
| `X.isel(lev=N)` | `FieldAtLevel` | `vertical_location` = `lev_N`; `N=0` → `model_top`, `N=-1` → `model_bot` |
| `X.interp(plev=V, units='hPa')` | `FieldAtPressureLevel` | `pressure_value`, `pressure_units` (default `Pa`) |
| `X.interp(z=V, reference='surface')` | `FieldAtHeight` | `height_value`, `height_units='m'`, `surface_reference` |
| `X.mean(dim='col')` | `HorizAvg` | `field_name` |
| `X.mean/sum(dim='lev')` | `VertContract` | `contract_method` = `avg`/`sum` |
| `X.min/max/std/var(dim='lev')` | `VertContract` | needs the hbc2 port |
| `X.weighted(W).mean(dim='lev')` | `VertContract` | `weighting_method` = `dp`/`dz` |
| `X.where(Y > 0)` | `ConditionalSampling` | `condition_lhs`, `condition_cmp`, `condition_rhs` |
| `X.shift(time=1)` | `FieldPrev` | `field_name` |
| `X.differentiate('p')` | `VertDerivative` | `derivative_method` |
| `X.histogram(bins=[a,b,c])` | `Histogram` | `bin_configuration` = `a_b_c` |
| `X.zonal_mean(bins=N)` | `ZonalAvg` | `number_of_zonal_bins` |
| `A + B`, `A - B`, `A * B`, `A / B` | `BinaryOp` | `arg1`, `arg2`, `binary_op` |
| `X / dt` | `FieldOverDt` | special-cased before `BinaryOp` |
| `log/exp/sqrt/abs(X)` | `UnaryOps` | needs the hbc2 port |
| `X.tend()` | expands to `(X - X.shift(time=1)) / dt` | built-in alias |

Notes on the awkward corners:

- **`.weighted()` is a modifier, not a diagnostic.** Mirroring xarray, it has
  no standalone meaning. Handle it inside the reduction handler: when walking
  `.mean(dim='lev')`, check whether the receiver is `X.weighted(W)`; if so,
  consume it, set `weighting_method`, and use `X` as `field_name`. Error
  clearly if `.weighted()` appears anywhere else.
- **`X / dt` must be checked before generic `BinaryOp`.** This is the same
  ordering constraint the regexes had — but now it is three lines with a
  comment in one function, keyed on the *identifier* `dt`, rather than an
  implicit dependency between two regexes. `dt` becomes a reserved identifier.
- **Comparisons are only legal inside `.where()`.** `T > 0` at top level is an
  error, not a diagnostic.
- **Unary minus has no home.** `-X` needs either a `neg` op on the ported
  `UnaryOps` diag, or lowering to `BinaryOp(X, times, -1)`. Prefer adding
  `neg`; note that BinaryOp's constant path goes through
  `physics::Constants<Real>::dictionary()`, which will not have `-1`.
- Constants (`Rgas`, `gravit`, ...) need no walker support: `BinaryOp` already
  resolves bare identifiers against the physics-constants dictionary.

### 8.6 Effect on the plan in §7

Steps 14-17 refine to:

- 14a. Add `AbstractDiagnostic::output_name` + the ~15 one-line diag edits.
  Standalone, no-op commit.
- 14b. Add the AST walker (`diag_from_ast.cpp`) with the args helper, the
  dispatch table, and the canonical-form idempotency property test. Unit
  tested without touching `create_diagnostic`.
- 15-17 as written.

## 9. Open decisions

- **Deprecation window for legacy names.** Indefinite support, or a stated
  release after which `legacy_diag_names.cpp` is deleted? This changes whether
  the shim is a permanent fixture or scaffolding.
- ~~**Do diags stop naming themselves?**~~ Resolved by §8.1: the optional
  `output_name` param gets us what the DSL needs for ~15 defaulted one-line
  edits, without the larger refactor. That larger refactor is no longer on the
  critical path at all.
- **`_bins` vs `groupby_bins`.** `X.zonal_mean(bins=20)` now, or go straight to
  the more xarray-faithful `X.groupby_bins('lat', 20).mean()`?
