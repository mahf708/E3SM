"""A general-purpose neural network over the columns this rank owns.

For any model whose output at a column depends on that column alone —
``y_i = f(x_i)`` — which covers pointwise parameterizations, column MLPs and
per-column vertical networks.  These are the easy case for scaling and it is
worth being explicit about why: the coupler already gave every rank a share of
the columns, the model has no receptive field beyond a column, so every rank
loads its own copy of the weights, runs its own batch and communicates
nothing.  Adding ranks adds throughput until the per-rank batch gets too small
to fill the device.

The hard case — a model whose *sample is the globe* — is not this; see
:mod:`e3sm_emulator.ace`.

Settings (``inference.*`` in the component namelist)::

    emulator:       generic
    model_path:     /path/to/model.pt   # TorchScript, or a pickled nn.Module
    input:          T                   # repeatable, ordered
    input:          q
    output:         dT
    dtype:          float32             # what the model was trained in
    device:         auto                # auto | cpu | cuda
"""

from __future__ import annotations

import numpy as np

from .context import Context


def build(config: dict, context: Context) -> "GenericEmulator":
    return GenericEmulator(config, context)


class GenericEmulator:
    """Runs one torch module on this rank's columns."""

    def __init__(self, config: dict, context: Context):
        import torch

        self.context = context
        self.verbose = bool(config.get("verbose", False))
        self.inputs = list(config.get("inputs") or [])
        self.outputs = list(config.get("outputs") or [])
        self.dtype = getattr(torch, str(config.get("dtype", "float32")))

        device = str(config.get("device", "auto")).lower()
        device_id = config.get("device_id")
        if device == "auto":
            self.device = context.torch_device(
                None if device_id in (None, "") else int(device_id)
            )
        else:
            self.device = torch.device(device)

        model_path = config.get("model_path") or ""
        if not model_path:
            raise ValueError(
                "The generic emulator needs `inference.model_path`, pointing "
                "at a TorchScript file or a pickled torch.nn.Module."
            )
        self.model = _load_module(model_path)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        if self.verbose and context.is_root:
            print(
                f"[e3sm_emulator.generic] {model_path} on {self.device}, "
                f"{len(self.inputs)} input(s) -> {len(self.outputs)} output(s)",
                flush=True,
            )

    def infer(self, inputs: dict, outputs: dict) -> None:
        import torch

        names_in = self.inputs or sorted(inputs)
        names_out = self.outputs or sorted(outputs)

        # Every field is [ncol, ...]; flatten each to [ncol, k] and lay them
        # out side by side, which is the layout a column network expects.
        columns = _column_count(inputs, names_in)
        packed = np.concatenate(
            [np.asarray(inputs[n]).reshape(columns, -1) for n in names_in], axis=1
        )

        with torch.no_grad():
            x = torch.from_numpy(packed).to(device=self.device, dtype=self.dtype)
            y = self.model(x)
        y = y.detach().to("cpu", torch.float64).numpy().reshape(columns, -1)

        # Split the result back over the named outputs, in order, using the
        # width each one declared by the array E3SM handed us.
        start = 0
        for name in names_out:
            target = outputs[name]
            width = int(np.asarray(target).size // columns) if columns else 0
            stop = start + width
            if stop > y.shape[1]:
                raise ValueError(
                    f"The model returned {y.shape[1]} values per column, which "
                    f"runs out while filling '{name}'. Check `inference.output` "
                    "against the model's actual output width."
                )
            target.reshape(columns, -1)[:] = y[:, start:stop]
            start = stop
        if start != y.shape[1]:
            raise ValueError(
                f"The model returned {y.shape[1]} values per column but the "
                f"declared outputs consume {start}."
            )


def _column_count(fields: dict, names) -> int:
    if not names:
        raise ValueError("No input fields were declared or supplied.")
    counts = {int(np.asarray(fields[n]).shape[0]) for n in names}
    if len(counts) != 1:
        raise ValueError(
            f"Input fields disagree about the column count: {sorted(counts)}."
        )
    return counts.pop()


def _load_module(path: str):
    """Load TorchScript if we can, otherwise a pickled module."""
    import torch

    try:
        return torch.jit.load(path, map_location="cpu")
    except Exception:
        # Not TorchScript. A checkpoint saved with torch.save(model) needs the
        # defining class importable, which is the user's business.
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.nn.Module):
            return obj
        raise ValueError(
            f"'{path}' is neither TorchScript nor a pickled torch.nn.Module "
            f"(it loaded as {type(obj).__name__}). Save the model with "
            "torch.jit.save, or point `inference.emulator` at an adapter that "
            "knows this format."
        )
