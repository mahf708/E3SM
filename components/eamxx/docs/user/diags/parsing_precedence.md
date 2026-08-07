# Legacy names

Before EAMxx parsed diagnostic requests as expressions, a diagnostic was
requested by writing a composite *name* — `T_mid_at_500hPa`,
`T_mid_vert_avg_dp_weighted`, `f_minus_f_prev_over_dt` — which was matched
against a list of patterns in a fixed order.

Those names still work, and still write identically-named output. They are
deprecated: new work should use the [expression syntax](dsl.md), which says
what it means instead of relying on the ordering rules below.

## How a request is resolved

1. **Plain model field.** If the name is a field in the model, it is used
   as-is and nothing is parsed.
2. **User `:=` alias.** Resolved to whatever it aliases.
3. **Parsed as an expression.** Anything that is not a single bare name is
   handled entirely by the grammar — see [Requesting diagnostics](dsl.md).
4. A single bare name is then resolved in this order:
   1. a registered diagnostic (`RelativeHumidity`, `SeaLevelPressure`, ...)
   2. a canonical named diagnostic (`LiqWaterPath`, `z_mid`, `dz`, ...)
   3. a legacy composite name, rewritten to the equivalent expression and
      re-parsed
   4. otherwise an error naming the unknown field or diagnostic

## Mapping

| Legacy name | Expression |
|---|---|
| `X_at_lev_10` | `X.isel(lev=10)` |
| `X_at_model_top` | `X.isel(lev=0)` |
| `X_at_model_bot` | `X.isel(lev=-1)` |
| `X_at_500hPa` | `X.interp(plev=500, units='hPa')` |
| `X_at_10m_above_surface` | `X.interp(z=10, reference='surface')` |
| `X_horiz_avg` | `X.mean(dim='col')` |
| `X_vert_avg` | `X.mean(dim='lev')` |
| `X_vert_sum` | `X.sum(dim='lev')` |
| `X_vert_avg_dp_weighted` | `X.weighted('dp').mean(dim='lev')` |
| `X_zonal_avg_20_bins` | `X.zonal_mean(bins=20)` |
| `X_where_Y_gt_0` | `X.where(Y > 0)` |
| `A_plus_B` | `A + B` |
| `A_minus_B` | `A - B` |
| `A_times_B` | `A * B` |
| `A_over_B` | `A / B` |
| `X_prev` | `X.shift(time=1)` |
| `X_over_dt` | `X / dt` |
| `X_atm_backtend` | `X.tend()` |
| `X_pvert_derivative` | `X.differentiate('p')` |
| `X_zvert_derivative` | `X.differentiate('z')` |
| `X_histogram_0_1_2` | `X.histogram(bins=[0,1,2])` |

## The precedence quirks, and why they are the reason to stop

Composite names are ambiguous, because the separator between parts is the same
`_` that appears inside field names. Three ordering rules were needed to
disambiguate them, and all three are still applied to legacy names so that they
keep meaning what they always meant.

**`_over_dt` is matched before binary ops.** Otherwise `X_over_dt` reads as
"X over dt", a division by a field named `dt`.

**The left operand is greedy, so the *rightmost* operator word is the outermost
operation.** `A_minus_B_over_C` means `(A − B) / C`. There is no way to write
`A − (B / C)` as a composite name at all. In the expression syntax you write
whichever you meant, and the parentheses are the answer.

**Binary ops are matched before `_prev`.** Otherwise `X_minus_X_prev` reads as
`FieldPrev(X_minus_X)` rather than `X − X_prev`.

None of this applies to expressions. `A - B / C` and `(A - B) / C` are
different requests, and both are expressible.

## Worked example

The backward-tendency family, in both spellings:

```yaml
fields:
  physics_pg2:
    aliases:
      # Intermediates: needed by the fields below, not written to the file
      - bt1 := T_mid.tend()
      - bt2 := bt1 - bt1.shift(time=1)
    field_names:
      - bt_prod      := bt1 * bt2
      - bt_osc_count := mask.where(bt_prod < 0)
```

The legacy equivalent, which still works:

```yaml
    aliases:
      - bt1 := T_mid_minus_T_mid_prev_over_dt
      - bt2 := bt1_minus_bt1_prev
    field_names:
      - bt_prod      := bt1_times_bt2
      - bt_osc_count := mask_where_bt_prod_lt_0
```

Note that `bt1` in the legacy form relies on `_over_dt` being matched first and
on the greedy left operand; the expression form does not rely on anything.
