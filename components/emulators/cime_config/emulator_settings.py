"""Write an emulated component's YAML input file for CIME.

Each emulated component reads `<class>_in` (atm_in, ocn_in, ice_in), a YAML
file naming its spec (components/emulators/specs) and the case's paths:

    spec: .../specs/samudra-e3smv3-ocean.yaml
    coupler_dt: 1800
    grid: {file: ..., domain: ocean_mask, mask_variable: mask_2d, publish_as: ocn}
    initial_condition: ...
    inference: {backend: libtorch, model_path: ..., device: cuda}

buildnml computes these from the case, and `user_nl_<component>` lines of
the form `dotted.key: value` override them (`inference.seed: 2027`).  A key
the component does not read is an error here, at case.setup, rather than a
run that stops at initialization.
"""

import os

import yaml

from CIME.utils import expect, safe_copy

# What EmulatorComponent reads (emulator_component.hpp).
KNOWN = {
    "spec": None,
    "coupler_dt": None,
    "initial_condition": None,
    "grid": {"file", "domain", "mask_variable", "publish_as", "shared_from"},
    "inference": {"backend", "model_path", "device", "dtype", "seed",
                  "jit_optimize", "num_threads"},
    "history": {"prefix", "interval", "fields"},
}


def source_specs(case):
    return os.path.join(case.get_value("SRCROOT"), "components", "emulators",
                        "specs")


def built_specs(case):
    return os.path.join(case.get_value("EXEROOT"), "emulators", "specs")


def snapshot_specs(case):
    """Copy the specs into EXEROOT at case.build.

    A spec names operators and keys the executable must understand, so a
    built case reads the specs it was built with: a spec edited in the source
    tree afterwards (a new operator key, say) would otherwise stop a
    continue run of an older executable at initialization.
    """
    src, dst = source_specs(case), built_specs(case)
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith(".yaml"):
            safe_copy(os.path.join(src, name), os.path.join(dst, name))


def spec_path(case, name):
    """The built snapshot's spec once the case is built, the source's before."""
    built = os.path.join(built_specs(case), name)
    return built if os.path.isfile(built) else os.path.join(source_specs(case),
                                                            name)


def read_user_settings(caseroot, compname, inst_string):
    """`dotted.key: value` lines from user_nl_<compname><inst_string>."""
    path = os.path.join(caseroot, "user_nl_{}{}".format(compname, inst_string))
    settings = {}
    if not os.path.isfile(path):
        return settings
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            expect(":" in text,
                   "{} line {}: expected `key: value`, got {!r}".format(
                       path, number, line.rstrip()))
            key, value = (part.strip() for part in text.split(":", 1))
            parts = key.split(".")
            known = parts[0] in KNOWN and (
                (KNOWN[parts[0]] is None and len(parts) == 1) or
                (KNOWN[parts[0]] is not None and len(parts) == 2 and
                 parts[1] in KNOWN[parts[0]]))
            expect(known,
                   "{} line {}: '{}' is not a setting the component reads; "
                   "known: {}".format(path, number, key, ", ".join(
                       k if v is None else "{}.{{{}}}".format(k, ",".join(sorted(v)))
                       for k, v in KNOWN.items())))
            settings[key] = yaml.safe_load(value)
    return settings


def write_input_file(case, caseroot, compname, class_name, settings):
    """Write <class>_in for every instance into Buildconf and RUNDIR."""
    rundir = case.get_value("RUNDIR")
    ninst = case.get_value("NINST_{}".format(class_name.upper())) or 1
    confdir = os.path.join(caseroot, "Buildconf", compname + "conf")
    os.makedirs(confdir, exist_ok=True)
    for inst in range(1, ninst + 1):
        inst_string = "_{:04d}".format(inst) if ninst > 1 else ""
        merged = {k: (dict(v) if isinstance(v, dict) else v)
                  for k, v in settings.items()}
        for key, value in read_user_settings(caseroot, compname,
                                             inst_string).items():
            parts = key.split(".")
            if len(parts) == 1:
                merged[parts[0]] = value
            else:
                merged.setdefault(parts[0], {})[parts[1]] = value
        if "history" in merged:
            # Named as the component's config_archive.xml expects.
            merged["history"].setdefault("prefix", "{}.{}{}.h".format(
                case.get_value("CASE"), compname, inst_string))
        filename = "{}_in{}".format(class_name, inst_string)
        path = os.path.join(confdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# {} input file, written by buildnml; change it in "
                    "user_nl_{}{}\n".format(compname, compname, inst_string))
            yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)
        if os.path.isdir(rundir):
            safe_copy(path, os.path.join(rundir, filename))
