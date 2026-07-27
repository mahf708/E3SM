"""Python emulators for E3SM's emulator components.

The C++ side of the bridge lives in
``components/emulators/common/src/inference``; it imports
:mod:`e3sm_emulator.bridge` and calls ``create_emulator(config)``.

Nothing here imports torch at module scope, so the decomposition and the
context can be exercised — and unit tested — on a machine with nothing but
numpy installed.
"""

from .context import Context

__all__ = ["Context"]
