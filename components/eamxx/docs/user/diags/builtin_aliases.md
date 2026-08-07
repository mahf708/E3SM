# Built-in aliases

Some requests are shorthand: rather than naming a diagnostic directly, they
expand to an equivalent expression, which is then resolved normally.

## Reference table

| Request | Expands to | Meaning |
|---|---|---|
| `X.tend()` | `(X - X.shift(time=1)) / dt` | Backward tendency of X |
| `X_atm_backtend` | `X.tend()` | Legacy spelling of the same thing |

## Example

```yaml
field_names:
  - dTdt := T_mid.tend()
```

which chains:

- `FieldPrev(T_mid)` → the value at the previous timestep
- `BinaryOp(T_mid, minus, ...)` → the difference
- `FieldOverDt(...)` → the difference divided by the timestep

The intermediates are computed and shared, but only `dTdt` is written to the
output file. If you want one of them in the file too, give it a name in the
`aliases:` section — see [IO aliases](../io_aliases.md).

## Notes

- Expansion is recursive: `X_atm_backtend` expands to `X.tend()`, which expands
  again. Anything a request expands to must itself be a valid expression.
- `.tend()` composes like any other operation, so
  `T_mid.tend().mean(dim='col')` is the column average of the tendency.
- To add an alias, return the expansion from `spec_from_ast` in
  `share/io/eamxx_diag_dsl.cpp` via `DiagSpec::rewrite_to`. Legacy name
  spellings live in `legacy_to_dsl` in `share/io/eamxx_diag_names.cpp`.
