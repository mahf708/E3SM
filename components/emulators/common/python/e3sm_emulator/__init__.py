"""Python emulators for E3SM's emulator components.

The C++ side imports :mod:`e3sm_emulator.bridge` and calls
``create_emulator(config)``; nothing here needs more than numpy.
"""

from .context import Context

__all__ = ["Context"]
