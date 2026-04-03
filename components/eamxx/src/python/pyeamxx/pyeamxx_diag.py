"""
EAMxx Python Diagnostic Framework
==================================

This module provides a @diagnostic decorator that lets climate scientists
define custom diagnostics in Python. Diagnostics can be either:

  1. JIT (Just-In-Time): Executed in Python at output time using NumPy.
     No compilation needed. Best for prototyping and moderate-frequency output.

  2. AOT (Ahead-Of-Time): The Python function is analyzed and translated
     to a C++ ExpressionDiag at case-setup time. Best for performance-critical
     diagnostics computed every timestep.

Usage:

    from pyeamxx_diag import diagnostic, Field, col_sum, col_avg, where

    @diagnostic(name="liquid_water_path", units="kg/m2")
    def lwp(qc: Field, pseudo_density: Field):
        return col_sum(qc * pseudo_density / 9.80616)

    @diagnostic(name="wind_speed", units="m/s")
    def ws(U: Field, V: Field):
        return sqrt(U**2 + V**2)

    @diagnostic(name="cloud_masked_T", units="K")
    def masked_T(T_mid: Field, cldfrac_liq: Field):
        return where(cldfrac_liq > 0.5, T_mid, -9999.0)

    @diagnostic(name="tropopause_T", units="K")
    def tropo_T(T_mid: Field, p_mid: Field):
        # Complex logic with loops and intermediates
        result = empty_like(T_mid[:, 0])  # 1D output (ncol)
        for icol in range(ncols):
            for k in range(nlevs):
                if p_mid[icol, k] < 10000.0:  # below 100 hPa
                    result[icol] = T_mid[icol, k]
                    break
        return result
"""

import inspect
import numpy as np
from typing import Callable, Optional, Dict, Any

# ============================================================
# Field proxy for symbolic tracing (AOT path)
# ============================================================

class Field:
    """Type annotation for diagnostic function parameters.
    In JIT mode, this is replaced by actual NumPy arrays.
    In AOT mode, this is a symbolic proxy for expression building."""
    pass


# ============================================================
# Reduction operations (work on NumPy arrays in JIT mode)
# ============================================================

def col_sum(field_2d):
    """Column sum: reduce over the last (level) dimension.
    Equivalent to vert_sum in EAMxx diagnostics."""
    return np.sum(field_2d, axis=-1)

def col_avg(field_2d):
    """Column average: mean over the last (level) dimension.
    Equivalent to vert_avg in EAMxx diagnostics."""
    return np.mean(field_2d, axis=-1)

def horiz_avg(field, area_weights=None):
    """Horizontal average: area-weighted mean over column dimension.
    Equivalent to horiz_avg in EAMxx diagnostics."""
    if area_weights is not None:
        return np.sum(field * area_weights, axis=0) / np.sum(area_weights)
    return np.mean(field, axis=0)

def where(condition, true_val, false_val):
    """Element-wise conditional: where(cond, a, b).
    Equivalent to conditional_sampling in EAMxx diagnostics."""
    return np.where(condition, true_val, false_val)

def empty_like(field):
    """Create an uninitialized array with the same shape."""
    return np.empty_like(field)


# ============================================================
# Diagnostic registry
# ============================================================

_registered_diagnostics: Dict[str, 'DiagnosticSpec'] = {}


class DiagnosticSpec:
    """Stores metadata and the callable for a registered diagnostic."""

    def __init__(self, func: Callable, name: str, units: str,
                 mode: str = "jit"):
        self.func = func
        self.name = name
        self.units = units
        self.mode = mode  # "jit" or "aot"

        # Introspect function signature to get field names
        sig = inspect.signature(func)
        self.input_field_names = list(sig.parameters.keys())

        # Try to extract a simple expression for AOT mode
        self.expression = None
        if mode == "aot":
            self.expression = self._try_extract_expression()

    def _try_extract_expression(self) -> Optional[str]:
        """Attempt to extract the function body as a single expression.
        This works for simple one-liner functions. For complex functions
        with loops/conditionals, returns None (must use JIT mode)."""
        source = inspect.getsource(self.func)
        lines = source.strip().split('\n')

        # Find the return statement
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('return '):
                expr = stripped[7:].strip()
                # Translate Python syntax to expression DSL syntax
                expr = expr.replace('**', '**')  # already compatible
                expr = expr.replace('np.', '')    # strip numpy prefix
                return expr
        return None

    def evaluate_jit(self, fields: Dict[str, np.ndarray]) -> np.ndarray:
        """Execute the diagnostic function in Python (JIT mode)."""
        # Build argument list from field names
        args = []
        for fname in self.input_field_names:
            if fname not in fields:
                raise ValueError(
                    f"Diagnostic '{self.name}' requires field '{fname}' "
                    f"but it was not provided. Available: {list(fields.keys())}")
            args.append(fields[fname])

        return self.func(*args)

    def to_yaml_derived_field(self) -> Optional[str]:
        """If the function can be expressed as a single expression,
        return the YAML derived_fields entry."""
        if self.expression:
            return f"{self.name} := {self.expression}"
        return None

    def __repr__(self):
        mode_str = "JIT" if self.mode == "jit" else "AOT"
        return (f"DiagnosticSpec(name='{self.name}', units='{self.units}', "
                f"mode={mode_str}, inputs={self.input_field_names})")


def diagnostic(name: str, units: str = "1", mode: str = "jit"):
    """Decorator to register a Python function as an EAMxx diagnostic.

    Parameters:
        name:  Output variable name in NetCDF files
        units: Physical units string (for metadata only in JIT mode)
        mode:  "jit" (execute in Python) or "aot" (translate to C++ expression)

    Example:
        @diagnostic(name="wind_speed", units="m/s")
        def ws(U, V):
            return sqrt(U**2 + V**2)
    """
    def decorator(func):
        spec = DiagnosticSpec(func, name, units, mode)
        _registered_diagnostics[name] = spec

        # Attach metadata to the function for introspection
        func._diag_spec = spec
        return func

    return decorator


def list_diagnostics():
    """List all registered Python diagnostics."""
    return dict(_registered_diagnostics)


def generate_yaml_block():
    """Generate the derived_fields YAML block for all AOT-compatible diagnostics."""
    lines = []
    for name, spec in _registered_diagnostics.items():
        entry = spec.to_yaml_derived_field()
        if entry:
            lines.append(f"      - {entry}")
        else:
            lines.append(f"      # {name}: too complex for AOT, requires JIT mode")
    return "    derived_fields:\n" + "\n".join(lines)


def evaluate_all(fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Evaluate all registered JIT diagnostics given a dict of input fields."""
    results = {}
    for name, spec in _registered_diagnostics.items():
        if spec.mode == "jit":
            try:
                results[name] = spec.evaluate_jit(fields)
            except Exception as e:
                print(f"Warning: diagnostic '{name}' failed: {e}")
    return results


# ============================================================
# Convenience math functions (delegate to numpy)
# ============================================================

sqrt = np.sqrt
abs = np.abs
log = np.log
log10 = np.log10
exp = np.exp


# ============================================================
# Example diagnostics (for documentation/testing)
# ============================================================

if __name__ == "__main__":
    # Example: define some diagnostics
    @diagnostic(name="wind_speed", units="m/s", mode="aot")
    def wind_speed(U, V):
        return sqrt(U**2 + V**2)

    @diagnostic(name="lwp", units="kg/m2", mode="jit")
    def lwp(qc, pseudo_density):
        return col_sum(qc * pseudo_density / 9.80616)

    @diagnostic(name="cloud_T", units="K", mode="jit")
    def cloud_T(T_mid, cldfrac_liq):
        return where(cldfrac_liq > 0.5, T_mid, np.nan)

    @diagnostic(name="T_anomaly", units="K", mode="aot")
    def T_anomaly(T_mid):
        return T_mid - 273.15

    # Show registered diagnostics
    print("Registered diagnostics:")
    for name, spec in list_diagnostics().items():
        print(f"  {spec}")
        yaml = spec.to_yaml_derived_field()
        if yaml:
            print(f"    -> YAML: {yaml}")

    # Generate YAML block
    print("\nGenerated YAML:")
    print(generate_yaml_block())

    # Evaluate JIT diagnostics on mock data
    print("\nEvaluating JIT diagnostics:")
    ncols, nlevs = 10, 72
    fields = {
        "U": np.random.randn(ncols, nlevs),
        "V": np.random.randn(ncols, nlevs),
        "T_mid": 250.0 + 30.0 * np.random.rand(ncols, nlevs),
        "qc": 1e-4 * np.random.rand(ncols, nlevs),
        "pseudo_density": 100.0 * np.ones((ncols, nlevs)),
        "cldfrac_liq": np.random.rand(ncols, nlevs),
    }

    results = evaluate_all(fields)
    for name, arr in results.items():
        print(f"  {name}: shape={arr.shape}, mean={arr.mean():.4g}")
