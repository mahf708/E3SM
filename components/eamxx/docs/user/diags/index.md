# Online diagnostics

EAMxx has facilities to output optional diagnostics
that are computed during runtime. These diagnostics
are designed generically and composably, and are requestable by users.

Diagnostics are requested by writing an expression in the output YAML, in a
syntax that follows Python and xarray:

```yaml
field_names:
  - T_vavg := T_mid.weighted('dp').mean(dim='lev')
  - cloudy := T_mid.where(qc > 1e-5)
```

Start with [Requesting diagnostics](dsl.md), which covers the whole syntax.

## Reference

- [Requesting diagnostics](dsl.md) — the expression syntax
- [Legacy names](parsing_precedence.md) — the deprecated `X_at_500hPa` style

## Individual diagnostics

- [Field contraction](field_contraction.md)
- [Conditional sampling](conditional_sampling.md)
- [Binary arithmetics](binary_ops.md)
- [Vertical derivative](vert_derivative.md)
- [Previous-timestep field](field_prev.md)
- [Field divided by timestep](field_over_dt.md)
- [Built-in aliases](builtin_aliases.md)

More details to follow.
