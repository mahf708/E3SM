# Expression diagnostics

Expression diagnostics allow you to define arbitrary arithmetic
computations on fields directly in your YAML output configuration,
without writing any C++ code. This is the most flexible way to
compute derived quantities in EAMxx.

## Overview

An expression diagnostic evaluates a mathematical expression
element-wise across all grid points and levels. You can use
field names, physics constants, numeric literals, arithmetic
operators, and mathematical functions.

Use the `derived_fields` key in your output YAML. Each entry
has the form `name := expression`, where `name` is the output
variable name that will appear in your NetCDF file:

```yaml
fields:
  physics_pg2:
    field_names:
      - T_mid
      - LiqWaterPath
    derived_fields:
      - my_water_path := qc * pseudo_density / gravit
      - wind_speed    := sqrt(U**2 + V**2)
      - T_anomaly     := T_mid - Tmelt
      - cloud_flag    := max(100 * (qc + qv) / gravit, 0.0) + log10(T_mid)
```

The `:=` syntax gives you control over the output variable name.
Each `derived_fields` entry produces a named field in the output
file, computed from the expression on the right-hand side.

> **Legacy syntax**: Expressions can also be requested via
> `field_names` using the `expr_` prefix (e.g., `expr_sqrt(U**2+V**2)`),
> but the `derived_fields` syntax is preferred because it gives you
> a clean output variable name instead of encoding the expression
> in the field name.

## Supported operators

| Operator | Description | Precedence |
| -------- | ----------- | ---------- |
| `+` | Addition | Low |
| `-` | Subtraction | Low |
| `*` | Multiplication | Medium |
| `/` | Division | Medium |
| `**` | Power (right-associative) | High |
| `-` (unary) | Negation | Highest |

Operator precedence follows standard mathematical conventions:
`a + b * c` is parsed as `a + (b * c)`.
Use parentheses to override: `(a + b) * c`.

## Supported functions

| Function | Arguments | Description |
| -------- | --------- | ----------- |
| `sqrt(x)` | 1 | Square root |
| `abs(x)` | 1 | Absolute value |
| `log(x)` | 1 | Natural logarithm |
| `exp(x)` | 1 | Exponential ($e^x$) |
| `square(x)` | 1 | Square ($x^2$) |
| `log10(x)` | 1 | Base-10 logarithm |
| `min(x, y)` | 2 | Element-wise minimum |
| `max(x, y)` | 2 | Element-wise maximum |

Functions can be nested: `sqrt(abs(x))`.

## Identifiers

### Field names

Any field registered in the EAMxx Field Manager can be used
by name. All referenced fields must have the same layout
(same dimensions and grid).

### Physics constants

Known physics constants are resolved automatically.
The full list of available constants includes:

| Constant | Description | Value | Units |
| -------- | ----------- | ----- | ----- |
| `gravit` | Gravitational acceleration | 9.80616 | m/s$^2$ |
| `Cpair` | Specific heat of dry air | 1004.64 | J/(kg K) |
| `Rair` | Gas constant for dry air | 287.042 | J/(kg K) |
| `LatVap` | Latent heat of vaporization | 2.501e6 | J/kg |
| `LatIce` | Latent heat of fusion | 3.337e5 | J/kg |
| `Tmelt` | Melting point of water | 273.15 | K |
| `P0` | Reference pressure | 100000 | Pa |
| `stebol` | Stefan-Boltzmann constant | 5.67e-8 | W/(m$^2$ K$^4$) |
| `Avogad` | Avogadro's number | 6.022e23 | 1/mol |
| `Boltz` | Boltzmann constant | 1.381e-23 | J/K |
| ... | | | |

Any identifier that is not a known physics constant or function
name is treated as a field name.

### Numeric literals

Numbers can be integers (`42`), decimals (`3.14`), or scientific
notation (`1.5e-3`, `2.0E+5`).

## Unit handling

Expression diagnostics output nondimensional units. The expression
engine does not track units through arbitrary arithmetic.
If you need unit-tracked output, use the binary or unary operations
diagnostics for simple cases.

## Limitations

- All referenced fields must have the **same layout** (same
  dimensions and same grid). You cannot mix 2D and 3D fields
  in one expression.
- The expression is evaluated **element-wise**. There is no
  support for reductions (sums, averages) within expressions.
  Use field contraction diagnostics for reductions.
- Maximum expression complexity: 64 instructions, 16 operand
  stack depth. This is sufficient for most practical expressions.
- Output units are always nondimensional.

## Example

```yaml
%YAML 1.1
---
filename_prefix: expression_outputs
averaging_type: instant
fields:
  physics_pg2:
    field_names:
      - T_mid
      - qv
    derived_fields:
      # Wind speed from components
      - wind_speed := sqrt(U**2 + V**2)

      # Mass-weighted mixing ratio
      - lwp := qc * pseudo_density / gravit

      # Temperature anomaly from freezing point
      - T_anomaly := T_mid - Tmelt

      # Saturation deficit (example)
      - sat_deficit := max(qv_sat - qv, 0.0)

      # Log-scale pressure
      - log_p := log10(p_mid)
output_control:
  frequency: 6
  frequency_units: nhours
```

## Relationship to other diagnostics

Expression diagnostics are the most general mechanism and
can replace most uses of binary and unary operations:

| Task | Binary/Unary syntax | Expression syntax |
| ---- | ------------------- | ----------------- |
| `a + b` | `a_plus_b` | `expr_a + b` |
| `a * c` | `a_times_c` | `expr_a * c` |
| `sqrt(x)` | `sqrt_of_x` | `expr_sqrt(x)` |
| `sqrt(a**2 + b**2)` | Not possible | `expr_sqrt(a**2 + b**2)` |

Use expression diagnostics when:

- You need more than one operation (e.g., `a * b + c`)
- You need nested functions (e.g., `sqrt(abs(x))`)
- You want natural mathematical notation

Use binary/unary operations when:

- You want **automatic unit tracking** in the output metadata
- You have a simple single-operation case
- You want backward compatibility with existing YAML files

## Python interface

The expression parser is also available from Python via the
`pyeamxx` module:

```python
import pyeamxx_ext

# Parse and inspect an expression
result = pyeamxx_ext.parse_expression("sqrt(u**2 + v**2)")
print(result.field_names)   # ['u', 'v']
print(len(result.program))  # 8 instructions

# Evaluate on NumPy arrays
import numpy as np
u = np.random.randn(100)
v = np.random.randn(100)
ws = pyeamxx_ext.eval_expression(
    "sqrt(u**2 + v**2)",
    {"u": u, "v": v}
)
```

This is useful for prototyping and validating expressions
before using them in production YAML configurations.
