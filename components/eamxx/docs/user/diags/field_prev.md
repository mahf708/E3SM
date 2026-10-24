# Previous-timestep field diagnostic

In EAMxx, we can capture the value of any field at the beginning of the
previous timestep. This is useful for comparing the state of the model
before and after a process updates a field within a single timestep.

## Configuration

To use this diagnostic, request the [expression](expressions.md)
`X.shift(time=1)`:

```yaml
field_names:
  - X_prev := X.shift(time=1)
```

The output field has the same layout, units, and grid as the input field `X`.

## Behavior

At the start of each timestep (during `init_timestep`), the current value of
`X` is saved. When the diagnostic is evaluated (during `compute_diagnostic`),
the saved value is written to output. This means `X.shift(time=1)` always reflects the
state of `X` at the beginning of the current timestep (i.e., before any
processes have modified it in the current step).

### First-step initialisation for derived diagnostics

When `X` is itself a computed diagnostic (rather than a raw model field) it
may not have been evaluated yet when `init_timestep` runs at the start of the
first step.  In that case the `init_timestep` call does nothing (the source
has no valid timestamp, so there is nothing to capture).

By the time `compute_diagnostic` is called — during the output phase, after
model physics and prerequisite diagnostics have been evaluated — `X` will have
a valid timestamp.  The diagnostic then uses the current value of `X` as a
stand-in for "X at t=0", producing:

```none
X.shift(time=1) = X(t=0)   →   X − X.shift(time=1) = 0 on the first step
```

This conservative initialisation prevents fill-value contamination from
propagating into downstream arithmetic (e.g. products of fill-value-sized
numbers overflow to Inf in single precision).

From the second step onward `X.shift(time=1)` reflects the normally captured
start-of-step value.

## Example

```yaml
field_names:
  # T_mid has units K
  - T_mid                          # current value of T_mid
  - T_mid_prev := T_mid.shift(time=1)  # value of T_mid at the start of the current timestep
```

## Caveats

- Only `time=1` is available, since only one step of history is kept.
- Only fields with `Real` data type are supported.

Contact developers on GitHub if you have additional questions.
