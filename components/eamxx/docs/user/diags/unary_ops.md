# Unary operations diagnostics

In EAMxx, we can apply mathematical functions to individual fields
to create new diagnostic outputs. The unary operations diagnostic
applies a single-argument function element-wise to a field.

## Supported operations

| Operator | Description | Unit requirement |
| -------- | ----------- | ---------------- |
| `sqrt` | Square root | Input must be nondimensional |
| `abs` | Absolute value | Any units (preserved) |
| `log` | Natural logarithm | Input must be nondimensional |
| `exp` | Exponential ($e^x$) | Input must be nondimensional |
| `square` | Square ($x^2$) | Any units (squared) |

## Configuration

To use the unary operations diagnostic, request an output
using the syntax `<op>_of_<field>`:

- `op` is the operator: `sqrt`, `abs`, `log`, `exp`, or `square`
- `field` is the name of the input field or a physics constant

## Unit handling

- `sqrt` and `log` and `exp` require the input to have nondimensional
  units. If you need `sqrt` of a dimensional field, first divide by
  appropriate units using the binary operations diagnostic.
- `abs` preserves the input units.
- `square` multiplies the input units by themselves
  (e.g., `m` becomes `m^2`).

## Physics constants

Like the binary operations diagnostic, you can use known
physics constant names (e.g., `gravit`, `Cpair`) instead of
field names. When a physics constant is used, the result is a
scalar diagnostic that is pre-computed once during initialization.

## Example

```yaml
%YAML 1.1
---
filename_prefix: unary_ops_outputs
averaging_type: instant
fields:
  physics_pg2:
    field_names:
      # abs preserves units
      - abs_of_omega

      # square multiplies units by themselves
      - square_of_T_mid

      # sqrt, log, exp require nondimensional input
      # (divide first if needed)
      - sqrt_of_o2mmr
output_control:
  frequency: 6
  frequency_units: nhours
```

## Relationship to other diagnostics

Unary operations are a simpler alternative to expression
diagnostics (see [Expression diagnostics](expressions.md))
when you only need a single function applied to a single field.
For multi-field or multi-operation expressions like
`sqrt(u**2 + v**2)`, use expression diagnostics instead.
