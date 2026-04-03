# Diagnostics system: developer guide

This document describes the EAMxx diagnostics architecture, how to
add new diagnostics, and the design decisions behind the system.
For user-facing documentation (YAML syntax), see
[user/diags/index.md](../user/diags/index.md).

## Architecture overview

```
User YAML                   Table-driven parser              Factory
field_names:                (eamxx_io_utils.cpp)             (register_diagnostics.hpp)
  - T_mid_at_500hPa   ---> regex match + param extract ---> AtmosphereDiagnosticFactory
  - expr_sqrt(u**2)         DiagSpec table                   .create("FieldAtPressureLevel",...)
  - LiqWaterPath                                             .create("ExpressionDiag",...)
                                                             .create("WaterPath",...)
```

The system has three layers:

1. **Name parser** (`eamxx_io_utils.cpp`): A table of `DiagSpec`
   structs, each with a regex pattern and a parameter-extraction
   lambda. Converts YAML field names into factory names + parameters.

2. **Factory** (`register_diagnostics.hpp`): A standard
   `ekat::Factory` that maps diagnostic names to constructor
   functions. Each diagnostic class registers itself here.

3. **Diagnostic classes** (`share/diagnostics/*.cpp`): Each class
   inherits from `AtmosphereDiagnostic`, declares its input fields,
   and implements `compute_diagnostic_impl()`.

### Base class hierarchy

```
AtmosphereProcess (share/atm_process/atmosphere_process.hpp)
  └── AtmosphereDiagnostic (share/atm_process/atmosphere_diagnostic.hpp)
        ├── WindSpeed
        ├── WaterPathDiagnostic
        ├── BinaryOpsDiag
        ├── UnaryOpsDiag
        ├── ExpressionDiag  ──uses──> diagnostics_core/
        ├── ConditionalSampling
        └── ... (31 total)
```

> **Design note**: `AtmosphereDiagnostic` inherits from
> `AtmosphereProcess`. This is acknowledged in the code as
> conceptually wrong (diagnostics are read-only observers, not
> state-mutating processes). It was done for convenience.
> The `diagnostics_core/` package provides a cleaner alternative
> base class (`ExpressionEvaluator`) that does not inherit from
> `AtmosphereProcess`. Future refactoring may decouple the two.

## Adding a new diagnostic: step by step

### Option 1: C++ diagnostic class (traditional)

This is the standard approach for diagnostics that need custom
Kokkos kernels (reductions, interpolation, etc.).

**Files to create/modify:**

1. `share/diagnostics/my_diag.hpp` -- class declaration
2. `share/diagnostics/my_diag.cpp` -- implementation
3. `share/diagnostics/register_diagnostics.hpp` -- factory registration
4. `share/io/eamxx_io_utils.cpp` -- YAML name pattern (DiagSpec table)
5. `share/diagnostics/CMakeLists.txt` -- add .cpp to library
6. `share/diagnostics/tests/my_diag_test.cpp` -- unit test
7. `share/diagnostics/tests/CMakeLists.txt` -- register test

**Step 1: Header file** (`my_diag.hpp`)

```cpp
#include "share/atm_process/atmosphere_diagnostic.hpp"

namespace scream {

class MyDiag : public AtmosphereDiagnostic {
public:
  MyDiag(const ekat::Comm& comm, const ekat::ParameterList& params);
  std::string name() const override { return "MyDiag"; }
  void create_requests() override;
protected:
  void compute_diagnostic_impl() override;
  // member variables for dimensions, cached values, etc.
};

} // namespace scream
```

**Step 2: Implementation** (`my_diag.cpp`)

```cpp
#include "my_diag.hpp"

namespace scream {

MyDiag::MyDiag(const ekat::Comm& comm, const ekat::ParameterList& params)
 : AtmosphereDiagnostic(comm, params)
{
  // Extract parameters from m_params
}

void MyDiag::create_requests() {
  auto grid = m_grids_manager->get_grid("physics");
  auto layout = grid->get_3d_scalar_layout(true);

  // Declare input fields
  add_field<Required>("T_mid", layout, K, grid->name());

  // Create and allocate the output field
  FieldIdentifier fid(name(), layout, units, grid->name());
  m_diagnostic_output = Field(fid);
  m_diagnostic_output.allocate_view();
}

void MyDiag::compute_diagnostic_impl() {
  // Kokkos kernel here
}

} // namespace scream
```

**Step 3: Register in factory** (`register_diagnostics.hpp`)

Add the include and registration:

```cpp
#include "my_diag.hpp"
// ...
diag_factory.register_product("MyDiag",
  &create_atmosphere_diagnostic<MyDiag>);
```

**Step 4: Add YAML name pattern** (`eamxx_io_utils.cpp`)

Add an entry to the `get_diag_specs()` table:

```cpp
{ R"(my_pattern_(\w+)$)",
  "MyDiag",
  [](const std::smatch& m,
     const std::shared_ptr<const AbstractGrid>& grid,
     ekat::ParameterList& p) {
    p.set("grid_name", grid->name());
    p.set<std::string>("some_param", m[1].str());
    return true;
  }
},
```

**Step 5: Add to CMake** (`diagnostics/CMakeLists.txt`)

```cmake
add_library(eamxx_diagnostics
  ...
  my_diag.cpp
)
```

**Step 6: Write a test** (`tests/my_diag_test.cpp`)

Follow the pattern in `binary_ops_test.cpp`: create a grid,
randomize input fields, run the diagnostic, and verify the output
against a manual calculation.

### Option 2: Expression diagnostic (no C++ needed)

For element-wise arithmetic on existing fields, users can define
diagnostics directly in YAML with no C++ changes:

```yaml
field_names:
  - expr_qc * pseudo_density / gravit
```

This works for any expression involving `+`, `-`, `*`, `/`, `**`,
`sqrt`, `abs`, `log`, `exp`, `square`, `min`, `max`, parentheses,
field names, physics constants, and numeric literals.

**When to use expressions vs. C++ diagnostics:**

| Expressions | C++ diagnostics |
| ----------- | --------------- |
| Element-wise arithmetic | Column reductions, interpolation |
| No recompilation needed | Full Kokkos kernel control |
| No unit tracking | Automatic unit propagation |
| Limited to same-layout fields | Can mix layouts |

### Option 3: Unary/binary operations (simple cases)

For single-operation cases where you want unit tracking:

```yaml
field_names:
  - sqrt_of_T_mid        # UnaryOpsDiag
  - qc_plus_qv           # BinaryOpsDiag
```

These are simpler than expressions and provide proper unit
metadata in the output NetCDF files.

## The DiagSpec table (name parser)

The `create_diagnostic()` function in `eamxx_io_utils.cpp` uses a
table-driven approach. Each entry is a `DiagSpec` struct:

```cpp
struct DiagSpec {
  std::string pattern_str;   // regex source
  std::string diag_name;     // factory key
  std::function<bool(const std::smatch&,
                     const std::shared_ptr<const AbstractGrid>&,
                     ekat::ParameterList&)> extract;
};
```

The table is defined in `get_diag_specs()` and evaluated in order.
The first matching pattern wins. To add a new pattern:

1. Append a new `DiagSpec` entry to the vector in `get_diag_specs()`.
2. The `extract` lambda populates the `ParameterList` with values
   extracted from regex match groups.
3. Order matters: more specific patterns should come before more
   general ones.

## The standalone package: `diagnostics_core/`

The expression engine has been extracted into a standalone,
model-agnostic package under `src/diagnostics_core/`. This
package has **zero EAMxx dependencies** (except for the adapter
header `eamxx_field_accessor.hpp`).

### Package structure

```
diagnostics_core/
  expression_parser.hpp       # Tokenizer + parser (std C++ only)
  rpn_evaluator.hpp           # GPU-safe Kokkos evaluator
  diagnostic_base.hpp         # ExpressionEvaluator<FieldAccessor>
  eamxx_field_accessor.hpp    # EAMxx adapter (only EAMxx dep)
  pydiag.hpp                  # Python/nanobind bindings
  CMakeLists.txt              # Header-only interface target
  tests/
    expression_parser_test.cpp
```

### FieldAccessor concept

To use the expression engine in a different model (MPAS, OMEGA,
etc.), implement this interface:

```cpp
struct MyModelFieldAccessor {
  using scalar_type = double;         // or float
  using field_type  = MyField;        // your field class

  // Return a device pointer to the field's flat data array
  static const scalar_type* get_data(const field_type& f);

  // Return a mutable device pointer
  static scalar_type* get_data_mut(field_type& f);

  // Return the total number of scalar elements
  static long long get_size(const field_type& f);

  // Return the field name (for error messages)
  static std::string get_name(const field_type& f);
};
```

Then instantiate the evaluator:

```cpp
using MyEvaluator = diag_utils::ExpressionEvaluator<MyModelFieldAccessor>;

MyEvaluator eval("sqrt(u**2 + v**2)", my_const_resolver);
eval.evaluate(input_fields, output_field);
```

### Expression parser internals

The parser is a recursive-descent parser that produces an RPN
(reverse Polish notation) instruction stream.

**Grammar:**

```
expr   = term (('+' | '-') term)*
term   = power (('*' | '/') power)*
power  = unary ('**' unary)*           [right-associative]
unary  = ('-' unary) | func '(' args ')' | atom
atom   = NUMBER | IDENTIFIER | '(' expr ')'
```

**Instruction set** (24 opcodes):

| Category | Opcodes |
| -------- | ------- |
| Push | PushField, PushConst |
| Binary arithmetic | Add, Sub, Mul, Div, Pow, Min, Max |
| Comparisons | CmpGt, CmpGe, CmpLt, CmpLe, CmpEq, CmpNe |
| Ternary | Where |
| Unary | Sqrt, Abs, Log, Log10, Exp, Square, Neg |

Comparisons return 1.0 (true) or 0.0 (false). `Where` pops three
values: condition, true_val, false_val; pushes true_val if
condition != 0, else false_val.

**Limits:** 64 instructions max, 16-element operand stack.

The `ConstResolver` callback allows models to plug in their own
physics constants:

```cpp
auto resolver = [](const std::string& name, double& val) -> bool {
  if (name == "gravity") { val = 9.81; return true; }
  return false;  // not a constant, treat as field name
};
auto result = diag_utils::parse_expression("x / gravity", resolver);
```

### RPN evaluator

The `evaluate_rpn()` function is marked `KOKKOS_INLINE_FUNCTION`
and runs entirely on fixed-size stack arrays. It is safe for CUDA,
HIP, and CPU execution spaces. The `Instruction` struct is trivially
copyable (POD) and can be stored in Kokkos Views.

## Conditional sampling implementation

The `ConditionalSampling` diagnostic uses a unified rank-agnostic
approach via flat data pointers. The single `apply_conditional_sampling()`
function handles both 1D and 2D fields, with either field-based or
level-index-based conditions.

Key parameters passed to the kernel:

- `use_lev_condition`: if true, condition value is the level index
- `nlevs_for_lev_cond`: controls how level indices are computed from
  the flat index (for 2D fields: `lev = flat_idx % nlevs`)

This replaced four separate near-identical functions, reducing code
from ~170 lines to ~75 lines.

## Python bindings

The expression parser is exposed to Python via nanobind. The
bindings are in `diagnostics_core/pydiag.hpp` and registered in
`src/python/libpyeamxx/pyeamxx_ext.cpp`.

Exposed Python API:

```python
# Parse an expression
result = pyeamxx_ext.parse_expression("sqrt(u**2 + v**2)")
result.field_names   # ['u', 'v']
result.program       # list of Instruction objects

# Parse with constant resolver
def resolve(name):
    if name == "g": return 9.81
    return None
result = pyeamxx_ext.parse_expression("x / g", resolve)

# Evaluate on numpy arrays
out = pyeamxx_ext.eval_expression("a + b", {"a": arr_a, "b": arr_b})
```

The Python evaluator runs on the host (not GPU). It is intended
for prototyping and validation, not production.

## Testing

Diagnostic tests live in `share/diagnostics/tests/`. Each test:

1. Creates a mesh-free grids manager
2. Creates input fields with randomized values
3. Instantiates the diagnostic via the factory
4. Calls `set_grids`, `set_required_field`, `initialize`, `compute_diagnostic`
5. Verifies output against a manual calculation

The standalone parser has its own tests in
`diagnostics_core/tests/expression_parser_test.cpp` that do not
depend on EAMxx infrastructure.

## Design decisions and trade-offs

### Why a table-driven parser instead of if-else chain?

The original `create_diagnostic()` was a ~150-line if-else chain of
regex matches. Adding a new diagnostic required inserting into the
middle of a fragile ordering-sensitive chain across multiple code
locations. The table-driven approach makes each pattern
self-documenting and independent.

### Why separate UnaryOpsDiag from ExpressionDiag?

`UnaryOpsDiag` provides proper unit tracking (e.g., `square(m)` = `m^2`).
`ExpressionDiag` always outputs nondimensional units because tracking
units through arbitrary arithmetic would require a full unit inference
engine. For simple single-function cases, `UnaryOpsDiag` is preferred.

### Why keep BinaryOpsDiag alongside ExpressionDiag?

Same reason: `BinaryOpsDiag` propagates units correctly
(`K * Pa` = `K*Pa`). It also predates `ExpressionDiag` and existing
YAML configurations depend on its naming convention. Both will
continue to work.

### Why RPN instead of an AST interpreter?

RPN (stack machine) has three advantages for GPU execution:

1. **No recursion**: fixed-size stack, no dynamic allocation
2. **No virtual dispatch**: a flat switch statement on POD opcodes
3. **Trivially copyable**: the instruction array can be stored in
   a Kokkos View and deep-copied to device memory

An AST interpreter would require heap-allocated tree nodes and
virtual dispatch, neither of which works on GPU.

### Why extract diagnostics_core as a separate package?

The expression engine has no inherent dependency on EAMxx. Other
Kokkos-based models (MPAS-Ocean, OMEGA, Parthenon) have the same
need for field-level arithmetic diagnostics. By separating the core
logic, it can be published as a standalone CMake package and consumed
via `FetchContent` or as a git submodule.

### Why nondimensional output for expressions?

Tracking units through arbitrary expressions (e.g., `sqrt(a*b + c/d)`)
would require building a parallel "unit expression tree" and evaluating
it symbolically. This is complex and error-prone. For simple cases,
use `BinaryOpsDiag` or `UnaryOpsDiag` which do track units. For
complex cases, the user is expected to know what units their expression
produces.

## Prototypes for advanced diagnostics

Two experimental approaches exist for diagnostics that go beyond
element-wise expressions (loops, intermediates, reductions).

### Prototype A: Python `@diagnostic` decorator

**File:** `src/python/pyeamxx/pyeamxx_diag.py`

Scientists write Python functions using NumPy, decorated with
`@diagnostic`:

```python
from pyeamxx_diag import diagnostic, col_sum, where, sqrt

@diagnostic(name="lwp", units="kg/m2")
def liquid_water_path(qc, pseudo_density):
    return col_sum(qc * pseudo_density / 9.80616)

@diagnostic(name="wind_speed", units="m/s", mode="aot")
def ws(U, V):
    return sqrt(U**2 + V**2)

@diagnostic(name="cloud_T", units="K")
def cloud_T(T_mid, cldfrac_liq):
    return where(cldfrac_liq > 0.5, T_mid, -9999.0)
```

**Two execution modes:**

- **JIT (default):** Function runs in Python via NumPy at output
  time. No compilation needed. Supports arbitrary logic (loops,
  conditionals, intermediates). Performance cost: Python overhead +
  host-device transfers. Acceptable for output-frequency diagnostics.

- **AOT:** Simple one-liner functions are analyzed and translated
  to `derived_fields` YAML entries. The `generate_yaml_block()`
  function emits the YAML. Complex functions that cannot be
  expressed as single expressions fall back to JIT.

**Key functions in `pyeamxx_diag`:**

| Function | Description |
| -------- | ----------- |
| `col_sum(x)` | Column sum (replaces `_vert_sum`) |
| `col_avg(x)` | Column average (replaces `_vert_avg`) |
| `horiz_avg(x)` | Horizontal average (replaces `_horiz_avg`) |
| `where(c, t, f)` | Conditional (replaces `_where_`) |
| `sqrt`, `abs`, `log`, `exp` | Math functions |

**When to use:** Complex logic with loops, intermediates, or
non-trivial conditionals that cannot be expressed as a single
arithmetic expression.

### Prototype B: Multi-statement compute blocks

**File:** `src/diagnostics_core/compute_block.hpp`

A YAML-native approach for multi-statement computations:

```yaml
compute_fields:
  - name: lwp
    inputs: [qc, pseudo_density]
    compute: |
      integrand = qc * pseudo_density / gravit
      result = col_sum(integrand)

  - name: cloud_masked_T
    inputs: [T_mid, cldfrac_liq]
    compute: |
      mask = gt(cldfrac_liq, 0.5)
      result = where(mask, T_mid, -9999.0)
```

Each line is `variable = expression` (element-wise) or
`variable = col_sum(expression)` (reduction). The special
variable `result` is the output.

**Supported statement types:**

| Statement | Description |
| --------- | ----------- |
| `var = expr` | Element-wise assignment |
| `var = col_sum(expr)` | Vertical sum reduction |
| `var = col_avg(expr)` | Vertical average reduction |
| `var = horiz_avg(expr)` | Horizontal average reduction |

**Status:** Parser is implemented in `compute_block.hpp`. The
evaluator is not yet integrated into the EAMxx output manager.
Integration would require:

1. Parsing `compute_fields` YAML entries in `scorpio_output.cpp`
2. A `ComputeBlockDiag` class that manages intermediate fields
   and dispatches to existing reduction implementations
3. Dependency tracking between statements

### Comparison of approaches

| Aspect | Expressions | Compute blocks | Python @diagnostic |
| ------ | ----------- | -------------- | ------------------ |
| Syntax | Single expression | Multi-line YAML | Python function |
| Intermediates | No | Yes | Yes |
| Loops | No | No | Yes |
| Reductions | No (compose) | col_sum, col_avg | col_sum, col_avg |
| Conditionals | where() | where() | Full Python if/else |
| GPU execution | Yes | Yes (planned) | No (host NumPy) |
| Compilation | None | None | None |
| Unit tracking | No | No | No |
| Replaces | BinaryOps, UnaryOps | ConditionalSampling, VertContract | All of the above |

### Recommended evolution path

1. **Now:** Use `derived_fields` with `where()`/comparisons for
   conditional masking. Compose with `_vert_sum` etc. suffixes
   for reductions.

2. **Near-term:** Integrate compute blocks for multi-statement
   diagnostics that need intermediates + reductions in one block.

3. **Medium-term:** Use Python `@diagnostic` for prototyping
   complex diagnostics. AOT-compatible ones auto-generate YAML.
   JIT handles the rest.

4. **Long-term:** Scientists write in Python, system auto-selects
   JIT (for complex logic) or AOT (for simple expressions)
   transparently.
