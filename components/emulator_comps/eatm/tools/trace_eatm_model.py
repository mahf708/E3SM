#!/usr/bin/env python
"""Trace an ACE-family checkpoint into the TorchScript file EATM loads.

This is a thin driver around the ACE repository's own tracing script: it reuses
that script's ``load_and_build`` (normalization + correctors + optional ocean
SST prescription, all folded into one nn.Module) and only takes over the final
``torch.jit.trace`` call.  Two reasons for that:

  * SamudrACE-E3SMv3's atmosphere is a *stochastic* NoiseConditionedSFNO.  It
    draws fresh noise on every forward pass, so ``torch.jit.trace``'s default
    ``check_trace=True`` re-runs the model, sees a different answer and raises.
    Tracing a stochastic module needs ``check_trace=False``.

  * EATM addresses channels by position, so the traced model's channel order
    has to match the table compiled into eatm_channels_mod.F90.  This script
    checks that before writing anything out, which turns a silent field mix-up
    into an error at trace time.

  * The ACE tracing script finds the corrector configuration at
    ``corrector._config``.  Current ``fme`` correctors do not expose that
    attribute -- the config lives at ``step.config.corrector`` -- and for a
    checkpoint with ``corrector_disabled_epochs`` the corrector is additionally
    wrapped in an ``EpochScheduledCorrector``.  The lookup therefore returns
    None and the script silently traces the bare network with no dry-air
    conservation, no moisture-budget closure and no force-positive clamping.
    This driver resolves the config from ``step.config.corrector`` instead and
    refuses to write a model whose correctors came out inactive when the
    checkpoint says they should be on.

Usage
-----
    python trace_eatm_model.py CHECKPOINT OUTPUT_BASE \\
        --emulator SamudrACE-E3SMv3 --device cuda

Writes ``OUTPUT_BASE_<device>.pt`` and ``OUTPUT_BASE_<device>_metadata.yaml``.
Point ``eatm_model_file`` in user_nl_eatm at the ``.pt``.

The checkpoint must be a single-component (atmosphere) stepper.  To get one out
of the coupled SamudrACE tar, use the ACE repo's
``scripts/coupled/create_decoupled_checkpoint.py --component atmosphere``, or
just download ``SamudrACE-E3SMv3-atmosphere.tar`` from Hugging Face.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import pathlib
import sys

import yaml

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Channel layouts EATM knows about.  Keep in sync with
# components/emulator_comps/eatm/src/eatm_channels_mod.F90.
def _levels(prefix: str) -> list[str]:
    return [f"{prefix}{k}" for k in range(8)]


EXPECTED: dict[str, dict[str, list[str]]] = {
    "ACE2-EAMv3": {
        "in": (
            ["LANDFRAC", "OCNFRAC", "ICEFRAC", "PHIS", "SOLIN", "PS", "TS"]
            + _levels("T_")
            + _levels("specific_total_water_")
            + _levels("U_")
            + _levels("V_")
        ),
        "out": (
            ["PS", "TS"]
            + _levels("T_")
            + _levels("specific_total_water_")
            + _levels("U_")
            + _levels("V_")
            + [
                "LHFLX",
                "SHFLX",
                "surface_precipitation_rate",
                "surface_upward_longwave_flux",
                "FLUT",
                "FLDS",
                "FSDS",
                "surface_upward_shortwave_flux",
                "top_of_atmos_upward_shortwave_flux",
                "tendency_of_total_water_path_due_to_advection",
            ]
        ),
    },
    "SamudrACE-E3SMv3": {
        "in": (
            ["LANDFRAC", "OCNFRAC", "ICEFRAC", "PHIS", "SOLIN", "PS", "TS"]
            + _levels("T_")
            + _levels("STW_")
            + _levels("U_")
            + _levels("V_")
            + ["Qat2m", "Uat10m", "Vat10m", "Tat2m"]
        ),
        "out": (
            ["PS", "TS"]
            + _levels("T_")
            + _levels("STW_")
            + _levels("U_")
            + _levels("V_")
            + [
                "LHFLX",
                "SHFLX",
                "surface_precipitation_rate",
                "frozen_precipitation_rate",
                "FLUS",
                "FLUT",
                "FLDS",
                "FSDS",
                "FSUS",
                "FSUTOA",
                "DTENDTTW",
                "TAUX",
                "TAUY",
                "Qat2m",
                "Uat10m",
                "Vat10m",
                "Tat2m",
            ]
        ),
    },
}


def _load_ace_tracer(path: pathlib.Path):
    """Import the ACE repository's tracing script as a module."""
    if not path.is_file():
        raise SystemExit(
            f"ERROR: ACE tracing script not found at {path}.\n"
            "Pass --trace-script, or set EATM_TRACE_SCRIPT."
        )
    spec = importlib.util.spec_from_file_location("ace_trace_script", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: could not import {path} as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ace_trace_script"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "load_and_build"):
        raise SystemExit(
            f"ERROR: {path} has no load_and_build(); this driver expects the "
            "corrector-aware tracing script, not the bare pt2ts.py helper."
        )
    return module


def _refresh_field_prefixes(tracer) -> None:
    """Give the tracing script fme's current field-name prefix map.

    The script inlines its own copy of ``ATMOSPHERE_FIELD_NAME_PREFIXES`` so
    the traced model has no runtime dependency on ``fme``.  That copy predates
    the SamudrACE naming: it knows ``specific_total_water_`` but not ``STW_``,
    and ``tendency_of_total_water_path_due_to_advection`` but not ``DTENDTTW``.
    The result is an empty water-channel list, and the dry-air corrector then
    dies at trace time with "stack expects a non-empty TensorList".
    """
    try:
        from fme.core.atmosphere_data import (  # noqa: PLC0415
            ATMOSPHERE_FIELD_NAME_PREFIXES as LIVE,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not import fme's field prefix map (%s); "
                       "using the tracing script's inlined copy", exc)
        return

    merged = dict(tracer.ATMOSPHERE_FIELD_NAME_PREFIXES)
    added = []
    for standard, prefixes in LIVE.items():
        old = merged.get(standard, [])
        new = list(prefixes) + [p for p in old if p not in prefixes]
        if new != old:
            added.extend(p for p in prefixes if p not in old)
        merged[standard] = new
    tracer.ATMOSPHERE_FIELD_NAME_PREFIXES = merged
    if added:
        logger.info("refreshed field-name prefixes from fme, adding: %s",
                    ", ".join(sorted(set(added))))


def _install_corrector_repair(tracer, holder: dict):
    """Make the ACE tracing script find the corrector config again.

    Replaces its ``_build_corrector_flags_and_channel_map`` with one that reads
    ``step.config.corrector`` (an ``AtmosphereCorrectorConfig``) rather than
    ``corrector._config``, which current ``fme`` correctors do not have.  The
    corrector *maths* is untouched -- only where the flags come from.

    The resolved config is stashed in ``holder`` so the caller can also rebuild
    the force-positive indices, which ``load_and_build`` looks up inline from
    the same dead attribute.
    """
    original = tracer._build_corrector_flags_and_channel_map

    def patched(stepper, in_names, all_out_names):
        step = tracer._unwrap_step(stepper)
        config = getattr(getattr(step, "config", None), "corrector", None)

        if config is None or not hasattr(config, "conserve_dry_air"):
            logger.warning(
                "could not resolve an atmosphere corrector config; falling "
                "back to the tracing script's own lookup"
            )
            return original(stepper, in_names, all_out_names)

        holder["config"] = config

        flags = {
            "conserve_dry_air": config.conserve_dry_air,
            "zero_global_mean_moisture_advection": (
                config.zero_global_mean_moisture_advection
            ),
            "moisture_budget_correction": config.moisture_budget_correction,
            "total_energy_budget_correction": (
                config.total_energy_budget_correction is not None
            ),
        }
        flags["any_active"] = any(
            [
                config.conserve_dry_air,
                config.zero_global_mean_moisture_advection,
                config.moisture_budget_correction is not None,
                config.total_energy_budget_correction is not None,
            ]
        )

        cmap = {
            "ps_in": tracer._find_by_standard(in_names, "surface_pressure"),
            "ps_out": tracer._find_by_standard(all_out_names, "surface_pressure"),
            "water_in_indices": tracer._find_all_by_standard(
                in_names, "specific_total_water"
            ),
            "water_out_indices": tracer._find_all_by_standard(
                all_out_names, "specific_total_water"
            ),
            "advection_out": tracer._find_by_standard(
                all_out_names, "tendency_of_total_water_path_due_to_advection"
            ),
            "precip_out": tracer._find_by_standard(all_out_names, "precipitation_rate"),
            "evap_out": tracer._find_by_standard(all_out_names, "latent_heat_flux"),
        }

        logger.info("resolved corrector config from step.config.corrector: %s", flags)

        # An unresolved index is not survivable: an empty water-channel list
        # makes the dry-air corrector fail inside torch.jit.trace with "stack
        # expects a non-empty TensorList", and a -1 silently addresses the last
        # channel.  Only complain about the ones the active correctors use.
        needed = {"ps_in", "ps_out"}
        if flags["conserve_dry_air"] or flags["moisture_budget_correction"]:
            needed |= {"water_in_indices", "water_out_indices"}
        if flags["moisture_budget_correction"]:
            needed |= {"precip_out", "evap_out", "advection_out"}
        if flags["zero_global_mean_moisture_advection"]:
            needed |= {"advection_out"}

        unresolved = [
            k for k in sorted(needed) if cmap[k] == -1 or cmap[k] == []
        ]
        if unresolved:
            holder["unresolved"] = unresolved
            logger.error(
                "these corrector channels did not resolve: %s", ", ".join(unresolved)
            )
        return flags, cmap

    tracer._build_corrector_flags_and_channel_map = patched


def _repair_force_positive(traceable, metadata: dict, holder: dict) -> None:
    """Rebuild the force-positive channel indices from the resolved config."""
    config = holder.get("config")
    names = list(getattr(config, "force_positive_names", []) or [])
    if not names:
        return

    all_out = [
        metadata["output_channels"][i] for i in range(len(metadata["output_channels"]))
    ]
    idx = torch.tensor(
        [all_out.index(n) for n in names if n in all_out], dtype=torch.long
    )
    if idx.numel() == traceable.force_pos_idx.numel():
        return  # the script already found them

    device = traceable.force_pos_idx.device
    traceable.force_pos_idx = idx.to(device)
    metadata["force_positive_names"] = names
    logger.info("restored %d force-positive channels: %s", idx.numel(), ", ".join(names))


def _check_channels(metadata: dict, emulator: str) -> None:
    expected = EXPECTED.get(emulator)
    if expected is None:
        logger.warning(
            "no channel expectations recorded for emulator %r; skipping check",
            emulator,
        )
        return

    n_in = metadata["n_input_channels"]
    got_in = [metadata["input_channels"][i] for i in range(n_in)]
    got_out = [
        metadata["output_channels"][i] for i in range(len(metadata["output_channels"]))
    ]

    problems = []
    if got_in != expected["in"]:
        problems.append(_diff("input", got_in, expected["in"]))
    if got_out != expected["out"]:
        problems.append(_diff("output", got_out, expected["out"]))

    if problems:
        raise SystemExit(
            "ERROR: the checkpoint's channel layout does not match the "
            f"'{emulator}' table in eatm_channels_mod.F90.\n"
            + "\n".join(problems)
            + "\nEATM addresses channels by position, so it would silently read "
            "the wrong fields.  Update the Fortran table (and the EXPECTED dict "
            "here) before using this checkpoint."
        )

    logger.info(
        "channel layout matches '%s' (%d in, %d out)", emulator, n_in, len(got_out)
    )


def _diff(kind: str, got: list[str], want: list[str]) -> str:
    lines = [f"  {kind} channels: got {len(got)}, expected {len(want)}"]
    for i in range(max(len(got), len(want))):
        g = got[i] if i < len(got) else "<missing>"
        w = want[i] if i < len(want) else "<missing>"
        if g != w:
            lines.append(f"    [{i}] got {g!r}, expected {w!r}")
    return "\n".join(lines[:25])


def main() -> None:
    default_script = os.environ.get(
        "EATM_TRACE_SCRIPT", "/pscratch/sd/m/mahf708/test_ace_repo/trace.py"
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", help="single-component ACE checkpoint (.tar/.ckpt)")
    parser.add_argument("output", nargs="?", default="eatm_traced", help="output base name")
    parser.add_argument(
        "--emulator",
        default="SamudrACE-E3SMv3",
        choices=sorted(EXPECTED),
        help="EATM_EMULATOR value this model will be used with",
    )
    parser.add_argument("--device", default="cuda", help="cpu or cuda (default: cuda)")
    parser.add_argument(
        "--trace-script",
        default=default_script,
        help="path to the ACE repository's corrector-aware tracing script",
    )
    parser.add_argument(
        "--add-ocean",
        action="store_true",
        help="fold the checkpoint's prescribed-SST step into the graph.  EATM "
        "blends the coupler's surface temperature in on the input side "
        "instead, so leave this off unless you also set eatm_pass_forcing.",
    )
    parser.add_argument(
        "--check-trace",
        action="store_true",
        help="re-run the model to verify the trace.  Only valid for a "
        "deterministic checkpoint; a NoiseConditionedSFNO will fail.",
    )
    parser.add_argument(
        "--allow-no-correctors",
        action="store_true",
        help="write the model even if the correctors came out inactive.  Only "
        "for a checkpoint that genuinely configures none.",
    )
    args = parser.parse_args()

    tracer = _load_ace_tracer(pathlib.Path(args.trace_script))

    from fme.core.distributed import Distributed  # noqa: PLC0415  (needs fme on path)

    holder: dict = {}
    _refresh_field_prefixes(tracer)
    _install_corrector_repair(tracer, holder)

    with Distributed.context():
        traceable, metadata = tracer.load_and_build(
            args.checkpoint,
            device=args.device,
            include_normalization=True,
            include_corrector=True,
            include_ocean=args.add_ocean,
        )

        _check_channels(metadata, args.emulator)
        _repair_force_positive(traceable, metadata, holder)

        config = holder.get("config")
        if config is not None and getattr(config, "clip_frozen_precipitation", None):
            logger.warning(
                "this checkpoint configures clip_frozen_precipitation, which "
                "the ACE tracing script does not implement"
            )

        if holder.get("unresolved"):
            raise SystemExit(
                "ERROR: the active correctors reference channels that could not "
                "be resolved: " + ", ".join(holder["unresolved"]) + ".\n"
                "The tracing script matches channels through "
                "ATMOSPHERE_FIELD_NAME_PREFIXES; this checkpoint uses names it "
                "does not know."
            )

        if not metadata["corrector_flags"].get("any_active"):
            message = (
                "the traced model has NO active correctors (no dry-air "
                "conservation, no moisture-budget closure).  EATM runs the "
                "emulator autoregressively for years, so an uncorrected model "
                "will drift."
            )
            if args.allow_no_correctors:
                logger.warning("%s  Continuing because --allow-no-correctors.", message)
            else:
                raise SystemExit(f"ERROR: {message}\nPass --allow-no-correctors to override.")

        if metadata["corrector_flags"].get("total_energy_budget_correction"):
            logger.warning(
                "this checkpoint configures a total energy budget corrector, "
                "which the ACE tracing script does not implement -- the traced "
                "model will not reproduce reference inference exactly"
            )

        n_in = metadata["n_input_channels"]
        n_forcing = metadata["n_forcing_channels"]
        n_lat, n_lon = metadata["n_lat"], metadata["n_lon"]
        example = torch.randn(1, n_in + n_forcing, n_lat, n_lon, device=args.device)

        logger.info(
            "tracing (%d state + %d forcing channels, %dx%d grid, check_trace=%s)",
            n_in,
            n_forcing,
            n_lat,
            n_lon,
            args.check_trace,
        )
        with torch.no_grad():
            traced = torch.jit.trace(
                traceable, (example,), check_trace=args.check_trace
            )

        out = pathlib.Path(f"{args.output}_{args.device}")
        pt_path = out.with_suffix(".pt")
        meta_path = out.parent / (out.name + "_metadata.yaml")

        torch.jit.save(traced, str(pt_path))
        metadata["eatm_emulator"] = args.emulator
        metadata["eatm_pass_forcing"] = bool(args.add_ocean)
        with open(meta_path, "w") as fh:
            yaml.dump(metadata, fh, default_flow_style=False, sort_keys=False)

        logger.info("wrote %s", pt_path)
        logger.info("wrote %s", meta_path)
        logger.info("")
        logger.info("Set in user_nl_eatm:")
        logger.info("  eatm_model_file = '%s'", pt_path.resolve())
        if args.add_ocean:
            logger.info("  eatm_pass_forcing = .true.")
        logger.info("and ./xmlchange EATM_EMULATOR=%s", args.emulator)


if __name__ == "__main__":
    main()
