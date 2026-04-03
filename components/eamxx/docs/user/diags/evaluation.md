# Evaluation rules: precedence, order, and composability

This page documents how EAMxx diagnostics are parsed, evaluated,
and composed. Understanding these rules helps you write correct
YAML configurations and avoid subtle bugs.

## Operator precedence

The expression parser follows standard mathematical precedence.
Operations higher in the table bind more tightly.

| Precedence | Operators | Associativity | Example |
| ---------- | --------- | ------------- | ------- |
| 1 (highest) | Unary `-` | right | `-x` |
| 2 | `**` (power) | right | `a ** b ** c` = `a ** (b ** c)` |
| 3 | `*`, `/` | left | `a * b / c` = `(a * b) / c` |
| 4 (lowest) | `+`, `-` | left | `a + b - c` = `(a + b) - c` |

**Key rule**: multiplication and division bind tighter than
addition and subtraction, just like in standard math.

```yaml
derived_fields:
  # This is parsed as: a + (b * c), NOT (a + b) * c
  - result := a + b * c

  # Use parentheses to override:
  - result2 := (a + b) * c

  # Power is right-associative:
  # a ** b ** c = a ** (b ** c), NOT (a ** b) ** c
  - result3 := 2.0 ** 3.0 ** 2.0    # = 2^(3^2) = 2^9 = 512
```

## Function evaluation

Functions are always evaluated before their result participates
in surrounding arithmetic:

```yaml
derived_fields:
  # sqrt is applied to (U**2 + V**2), not just U
  - ws := sqrt(U**2 + V**2)

  # Nested functions: inner first, then outer
  - x := sqrt(abs(T_mid - 273.15))

  # Functions with multiple arguments use commas
  - y := min(T_mid, 300.0)
  - z := where(gt(cldfrac, 0.5), T_mid, -9999.0)
```

### Available functions

| Function | Args | Description |
| -------- | ---- | ----------- |
| `sqrt(x)` | 1 | Square root |
| `abs(x)` | 1 | Absolute value |
| `log(x)` | 1 | Natural logarithm |
| `log10(x)` | 1 | Base-10 logarithm |
| `exp(x)` | 1 | Exponential |
| `square(x)` | 1 | Square ($x^2$) |
| `min(a, b)` | 2 | Element-wise minimum |
| `max(a, b)` | 2 | Element-wise maximum |
| `gt(a, b)` | 2 | Greater than (returns 1.0 or 0.0) |
| `ge(a, b)` | 2 | Greater than or equal |
| `lt(a, b)` | 2 | Less than |
| `le(a, b)` | 2 | Less than or equal |
| `eq(a, b)` | 2 | Equal |
| `ne(a, b)` | 2 | Not equal |
| `where(c, t, f)` | 3 | If c != 0 then t, else f |

### Comparison and where: replacing conditional_sampling

The comparison functions (`gt`, `lt`, etc.) return `1.0` for true
and `0.0` for false. Combined with `where`, they replace the
older `conditional_sampling` syntax:

```yaml
# Old syntax (conditional_sampling):
field_names:
  - T_mid_where_cldfrac_liq_gt_0.5

# New syntax (expression with where):
derived_fields:
  - masked_T := where(gt(cldfrac_liq, 0.5), T_mid, -9999.0)
```

The expression syntax is more flexible:

```yaml
derived_fields:
  # Compound conditions (not possible with conditional_sampling)
  - warm_cloud_T := where(gt(cldfrac_liq, 0.5) * gt(T_mid, 273.15), T_mid, 0.0)

  # Custom fill values
  - masked_q := where(gt(qv, 1e-6), qv, 0.0)

  # Clamp to a range
  - T_clamped := min(max(T_mid, 200.0), 350.0)
```

## Evaluation order and composability

### How field names are resolved

When EAMxx processes an output YAML file, it resolves field names
in this order:

1. **Model fields**: Names matching fields already in the Field Manager
   (e.g., `T_mid`, `qv`, `p_mid`)
2. **Aliases**: Names with `:=` syntax that rename existing fields
   (e.g., `temperature:=T_mid`)
3. **Derived fields**: Entries in `derived_fields` that define
   expressions (e.g., `ws := sqrt(U**2 + V**2)`)
4. **Built-in diagnostics**: Names matching regex patterns in the
   diagnostic spec table (e.g., `LiqWaterPath`, `T_mid_at_500hPa`)

Within each category, dependencies are resolved iteratively.
A field that depends on another is deferred until its dependency
is available. This means you can chain diagnostics:

```yaml
fields:
  physics_pg2:
    field_names:
      # Step 1: model field T_mid is already available
      - T_mid

      # Step 3: uses the derived field from below
      - T_celsius_at_500hPa

    derived_fields:
      # Step 2: creates a field called T_celsius
      - T_celsius := T_mid - 273.15
```

In this example, `T_celsius` is created by the expression diagnostic,
then `T_celsius_at_500hPa` is resolved as a `FieldAtPressureLevel`
diagnostic applied to `T_celsius`.

### Composing with reductions

Derived fields create new entries in the Field Manager. You can
then apply any existing reduction suffix to them:

```yaml
fields:
  physics_pg2:
    derived_fields:
      # Create element-wise intermediate
      - q_weighted := qc * pseudo_density

    field_names:
      # Apply vertical sum to the derived field
      - q_weighted_vert_sum

      # Apply horizontal average to the derived field
      - q_weighted_horiz_avg

      # Apply zonal average
      - q_weighted_zonal_avg_36_bins

      # Conditional sample, then reduce
      - q_weighted_where_T_mid_lt_273.15
      - q_weighted_where_T_mid_lt_273.15_vert_sum
```

This composability means you do NOT need reductions built into
the expression language. Instead, define the element-wise
computation as a derived field, then use the existing reduction
suffixes.

### Composition order

When chaining multiple suffixes, they are applied **left-to-right**
as the parser strips them. For example:

| YAML name | Parsed as |
| --------- | --------- |
| `T_mid_vert_sum_horiz_avg` | horiz_avg( vert_sum( T_mid ) ) |
| `T_mid_horiz_avg_vert_sum` | vert_sum( horiz_avg( T_mid ) ) |
| `T_mid_where_qv_gt_0.01_vert_avg` | vert_avg( T_mid_where_qv_gt_0.01 ) |

**Warning**: not all compositions are meaningful. For example,
applying `vert_sum` after `horiz_avg` on a 2D field produces
a different result than applying `horiz_avg` after `vert_sum`.

### Dependency cycles

Circular dependencies are detected and will cause an error at
initialization time. For example, this is invalid:

```yaml
derived_fields:
  - a := b + 1.0
  - b := a + 1.0    # ERROR: circular dependency
```

## Internal evaluation: RPN stack machine

Expressions are compiled to a Reverse Polish Notation (RPN)
instruction stream and evaluated by a GPU-safe stack machine.
This is an implementation detail, but understanding it helps
debug complex expressions.

For example, `a + b * c` compiles to:

```
PUSH a    # stack: [a]
PUSH b    # stack: [a, b]
PUSH c    # stack: [a, b, c]
MUL       # stack: [a, b*c]
ADD       # stack: [a + b*c]
```

And `where(gt(x, 0), sqrt(x), 0)` compiles to:

```
PUSH x    # stack: [x]
PUSH 0    # stack: [x, 0]
CMP_GT    # stack: [x>0 ? 1.0 : 0.0]
PUSH x    # stack: [cond, x]
SQRT      # stack: [cond, sqrt(x)]
PUSH 0    # stack: [cond, sqrt(x), 0]
WHERE     # stack: [cond!=0 ? sqrt(x) : 0]
```

### Limits

| Limit | Value | What happens if exceeded |
| ----- | ----- | ----------------------- |
| Max instructions per expression | 64 | Parse error at initialization |
| Max operand stack depth | 16 | Undefined behavior (avoid deeply nested expressions) |
| Max fields per expression | 16 | Runtime error at initialization |

For most practical expressions, these limits are never reached.
An expression like `sqrt(U**2 + V**2)` uses 8 instructions and
stack depth 3.

## Choosing between approaches

| Task | Recommended approach | Why |
| ---- | -------------------- | --- |
| Simple arithmetic (`a + b`, `a * b`) | `derived_fields` or `field_names` with binary ops | Clean, unit-tracked |
| Single function (`sqrt(x)`) | `derived_fields` or `field_names` with unary ops | Unit-tracked for unary ops |
| Multi-field formula | `derived_fields` | Natural math syntax |
| Conditional masking | `derived_fields` with `where()` | Replaces conditional_sampling, more flexible |
| Vertical/horizontal reductions | `field_names` with `_vert_sum` etc. suffixes | Existing, well-tested |
| Reduction of a computed quantity | `derived_fields` + reduction suffix in `field_names` | Compose element-wise + reduction |
| Complex logic with loops | Python `@diagnostic` decorator (JIT) | Full Python expressiveness |
| Performance-critical custom kernel | C++ diagnostic class | Full Kokkos control |
