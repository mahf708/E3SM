# Diagnostic expressions

Alongside the underscore names described in the rest of this section, EAMxx can
build diagnostics from **expressions**:

```yaml
field_names:
  - q_flux := (qc+qv)*p_mid
  - cold_T  := T_mid.where(T_mid<273.15).mean('lev')
```

Expressions are parsed by [`dexpr`](https://github.com/E3SM-Project/E3SM/tree/master/share/dexpr),
the expression parser shared across E3SM components. They exist because the
underscore syntax cannot express grouping: `qc_plus_qv_times_p_mid` is grouped
by regex greediness, not by operator precedence (see
[Binary arithmetics](binary_ops.md) and [Parsing precedence](parsing_precedence.md)).
Written as an expression, it means what it looks like.

Both syntaxes work, and they are resolved independently: a name is matched
against the underscore patterns first, and only names that match none of them
are read as an expression. Nothing you write today changes meaning.

## Expressions must be given a name

An expression is not a usable NetCDF variable name, and `field_names` entries
are written to file verbatim. So any expression you want written must be given
a name with `:=`:

```yaml
field_names:
  - q_flux := (qc+qv)*p_mid    # good
  - (qc+qv)*p_mid              # error: "An expression must be given an output name"
```

The expression itself is recorded in the variable's `alias_of` attribute, so
the provenance stays in the file.

This applies only to what is *written*. Expressions nested inside other
expressions, and expressions in the `aliases:` section (intermediates that other
diagnostics use but that are not written), need no name of their own.

## Operators

| Operator | Meaning |
| -------- | ------- |
| `a + b`, `a - b`, `a * b`, `a / b` | element-wise arithmetic |
| `>`, `>=`, `<`, `<=`, `==`, `!=` | comparison, **only** inside `where(..)` |
| `( )` | grouping |

Operands may be fields, other expressions, named physical constants (`Rgas`,
`P0`, ...), or numeric literals:

```yaml
field_names:
  - p_hPa      := p_mid/100
  - T_scaled   := T_mid*0.5
  - dry_air    := p_mid - qv*p_mid
```

Precedence is the usual one, so `qc+qv*p_mid` is `qc+(qv*p_mid)`. Note this
differs from the underscore spelling `qc_plus_qv_times_p_mid`, which means
`(qc+qv)*p_mid`. When in doubt, parenthesize.

## Functions

Functions are written as methods on the field they act on, `X.f(..)`, and are
named after their [xarray](https://docs.xarray.dev) counterparts wherever there
is an honest analogue.

| Expression | Equivalent underscore name |
| ---------- | -------------------------- |
| `X.isel(lev=10)` | `X_at_lev_10` |
| `X.isel(lev=-1)` | `X_at_model_bot` |
| `X.isel(lev='model_top')` | `X_at_model_top` |
| `X.interp(plev=500, units='hPa')` | `X_at_500hPa` |
| `X.interp(z=10, reference='surface')` | `X_at_10m_above_surface` |
| `X.mean('col')` | `X_horiz_avg` |
| `X.mean('lev')` | `X_vert_avg` |
| `X.sum('lev')` | `X_vert_sum` |
| `X.mean('lev', weights='dp')` | `X_vert_avg_dp_weighted` |
| `X.where(qv>0.01)` | `X_where_qv_gt_0.01` |
| `X.differentiate('p')` | `X_pvert_derivative` |
| `X.histogram([0,1,2])` | `X_histogram_0_1_2` |
| `X.zonal_mean(bins=20)` | `X_zonal_avg_20_bins` |
| `X.prev()` | `X_prev` |
| `X.over_dt()` | `X_over_dt` |
| `X.tend()` | `X_atm_backtend` |

Notes:

- `isel` indexes levels the way xarray does, so `lev=-1` is the bottom level.
  Other negative indices are rejected, since they cannot be resolved without
  knowing the level count.
- `interp` takes exactly one of `plev` or `z`. `units` defaults to `Pa` for
  `plev` and `m` for `z`; `reference` (`'surface'` or `'sealevel'`, default
  `'sealevel'`) applies to `z` only.
- `mean('col')` is always area weighted, so `weights` does not apply to it.
  `weights` for `'lev'` is `'dp'` or `'dz'`.
- `where` takes a single comparison. `and`/`or` are not supported; chain
  `where(..)` calls instead. The special operand names `mask` and `lev` work as
  they do in the underscore syntax, so `mask.where(lev>5)` is valid.
- `X.tend()` is shorthand for `(X - X.prev()).over_dt()`.
- Histogram bin edges must be non-negative, and must be writable without an
  exponent: the diagnostic joins them with `_` and splits them back apart, so
  `1e3` is fine (it is written `1000.0`) but `1e30` is not.

Named quantities are not operations, and stay plain names in an expression:
`LiqWaterPath`, `RelativeHumidity`, `z_mid`, `dz`, `precip_liq_surf_mass_flux`,
and so on.

## Errors

An expression that does not parse, calls a function that does not exist, or
uses an operator no diagnostic implements is rejected with a message pointing
at the offending part. Unknown function names come back with the list of
available ones.

```text
Error! Unsupported expression: no diagnostic implements the operator '**'.
 - subexpression: (T_mid**2)
```

## Worked example

```yaml
fields:
  physics_pg2:
    aliases:
      # An intermediate: used below, but not written to file. No name needed
      # for the expression itself beyond the alias.
      - warm_T := T_mid.where(T_mid>273.15)

    field_names:
      # Column-mean temperature, but only where it is above freezing
      - warm_T_colavg := warm_T.mean('lev')
      # Fraction of the globe where the surface is warm
      - warm_frac     := mask.where(T_mid.isel(lev=-1)>273.15).mean('col')
      # Moisture flux
      - q_flux        := (qc+qv)*p_mid
```

## Adding a function

The set of callable functions is registered by EAMxx, in
`components/eamxx/src/share/io/eamxx_dexpr_diags.cpp`, not by `dexpr` itself.
Adding one means adding a `FunctionSpec` to the registry there and a case to the
translator that maps it onto a diagnostic; nothing under `share/dexpr` changes.
