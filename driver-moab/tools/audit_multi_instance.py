#!/usr/bin/env python3
"""Audit driver-moab for single-instance (NINST == 1) assumptions.

cpl7/mct supports CIME's multi-instance capability (NINST_<COMP> > 1); cpl7/moab
does not.  This script inventories the places that hard-code a single instance so
the porting effort can be scoped and tracked.  It is a static audit: it only
greps the sources, it does not build or run the model.

Usage:
    ./driver-moab/tools/audit_multi_instance.py [--verbose]

Exit status is 0 unless an expected source file is missing.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.dirname(HERE)
ROOT = os.path.dirname(DRIVER)

# Coupler-side (and component-side) iMOAB application handles.  In driver-mct the
# equivalent state lives in component_type_mod's atm(:)/lnd(:)/... arrays, which
# are already dimensioned by num_inst_*.  These are plain scalars, so a second
# instance would overwrite the first.
APP_ID_VARS = [
    # component-side handles (set by the component, read by the driver)
    "mhid", "mhfid", "mhpgid", "mphaid", "mpoid", "mlnid", "mrofid", "mpsiid",
    # coupler-side mesh handles
    "mbaxid", "mboxid", "mbofxid", "mblxid", "mbixid", "mbrxid",
    # intersection meshes / remap-weight holders
    "mbintxao", "mbintxoa", "mbintxla", "mbintxal", "mbintxia",
    "mbintxro", "mbintxor", "mbintxar", "mbintxlr", "mbintxrl", "mbintxri",
]

COMPS = ["atm", "lnd", "ocn", "ice", "rof", "glc", "wav", "iac"]

# Component sources that pin an iMOAB application id to instance 1.
COMPONENT_FILES = [
    "components/eam/src/dynamics/se/dyn_comp.F90",
    "components/eamxx/src/dynamics/homme/interface/phys_grid_mod.F90",
]


def fortran_sources(directory):
    out = []
    for base, _, files in os.walk(directory):
        for name in sorted(files):
            if name.endswith(".F90"):
                out.append(os.path.join(base, name))
    return sorted(out)


def read(path):
    with open(path, "r", errors="replace") as handle:
        return handle.read()


def section(title):
    print()
    print(title)
    print("-" * len(title))


def audit_app_ids(verbose):
    """Module-level scalars that hold iMOAB application handles."""
    decl_file = os.path.join(DRIVER, "shr", "seq_comm_mct.F90")
    decls = read(decl_file)
    sources = fortran_sources(os.path.join(DRIVER, "main"))
    sources.append(decl_file)

    # Collect every name declared as a bare integer scalar, allowing the
    # comma-separated form "integer, public :: mhid, mhfid, mpoid, mlnid".
    declared_scalars = set()
    for decl in re.findall(r"^[ \t]*integer[ \t]*,[ \t]*public[ \t]*::[ \t]*([^!\n]+)",
                           decls, re.M):
        for name in decl.split(","):
            name = name.strip()
            if re.fullmatch(r"[A-Za-z_]\w*", name):
                declared_scalars.add(name.lower())

    counts = {}
    for var in APP_ID_VARS:
        pattern = re.compile(r"\b%s\b" % re.escape(var), re.I)
        total = sum(len(pattern.findall(read(src))) for src in sources)
        counts[var] = (var.lower() in declared_scalars, total)

    section("1. iMOAB application handles declared as scalars (need rank-1 by instance)")
    scalars = [v for v, (is_scalar, _) in counts.items() if is_scalar]
    refs = sum(total for v, (is_scalar, total) in counts.items() if is_scalar)
    print("   %d of %d handles are scalars, %d references in driver-moab"
          % (len(scalars), len(APP_ID_VARS), refs))
    if verbose:
        for var in APP_ID_VARS:
            is_scalar, total = counts[var]
            print("     %-10s scalar=%-5s refs=%d" % (var, is_scalar, total))
    return len(scalars), refs


def audit_init_guard():
    """cplcomp_moab_Init is called for instance 1 only."""
    path = os.path.join(DRIVER, "main", "component_mod.F90")
    text = read(path)
    guarded = re.search(
        r"if\s*\(\s*eci\s*==\s*1\s*\)\s*then.*?cplcomp_moab_Init\s*\(", text, re.S
    ) is not None
    section("2. Coupler-side MOAB mesh is built for instance 1 only")
    print("   component_mod.F90 calls cplcomp_moab_Init under 'if (eci == 1)': %s"
          % guarded)
    print("   -> instances 2..N never get a coupler-side mesh or field tags.")
    return guarded


def audit_hardcoded_instance():
    """comp(1) appearing on lines that also touch the MOAB data path."""
    moab_marker = re.compile(r"iMOAB|_moab|mb[a-z]*xid|mphaid|mpoid|mlnid|mrofid|mpsiid")
    inst1 = re.compile(r"\b(?:%s)\(1\)" % "|".join(COMPS))
    hits = []
    for src in fortran_sources(os.path.join(DRIVER, "main")):
        for lineno, line in enumerate(read(src).splitlines(), 1):
            if line.lstrip().startswith("!"):
                continue
            if inst1.search(line) and moab_marker.search(line):
                hits.append((os.path.relpath(src, ROOT), lineno, line.strip()))
    section("3. MOAB calls hard-coded to instance 1")
    print("   %d call sites pass <comp>(1) into a MOAB routine" % len(hits))
    for rel, lineno, line in hits:
        print("     %s:%d: %s" % (rel, lineno, line[:110]))
    return len(hits)


def audit_component_side():
    """Component sources that register an iMOAB app with instance 1's id."""
    section("4. Component-side iMOAB app ids pinned to instance 1")
    hits = []
    for rel in COMPONENT_FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print("   MISSING: %s" % rel)
            return None
        for lineno, line in enumerate(read(path).splitlines(), 1):
            if re.search(r"\b(?:ATM|LND|OCN|ICE|ROF|GLC|WAV)ID\(1\)", line):
                hits.append((rel, lineno, line.strip()))
    print("   %d registrations use <COMP>ID(1) instead of the running instance"
          % len(hits))
    for rel, lineno, line in hits:
        print("     %s:%d: %s" % (rel, lineno, line[:110]))
    return len(hits)


def audit_guards():
    """Fail-fast guards already present in the driver."""
    path = os.path.join(DRIVER, "main", "cime_comp_mod.F90")
    text = read(path)
    section("5. Fail-fast guards in driver-moab")
    multi_driver = "Multi-driver not supported" in text
    multi_inst = "Multi-instance not supported" in text
    print("   MULTI_DRIVER > 1 aborts:        %s" % multi_driver)
    print("   NINST_<COMP> > 1 aborts:        %s" % multi_inst)
    return multi_driver, multi_inst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="list every application handle and its reference count")
    args = parser.parse_args()

    for required in [os.path.join(DRIVER, "shr", "seq_comm_mct.F90"),
                     os.path.join(DRIVER, "main", "component_mod.F90"),
                     os.path.join(DRIVER, "main", "cime_comp_mod.F90")]:
        if not os.path.exists(required):
            sys.stderr.write("ERROR: missing %s\n" % required)
            return 1

    print("driver-moab multi-instance readiness audit")
    print("=========================================")

    audit_app_ids(args.verbose)
    audit_init_guard()
    audit_hardcoded_instance()
    if audit_component_side() is None:
        return 1
    audit_guards()

    print()
    print("driver-moab is single-instance by construction; see")
    print("docs/dev-guide/moab-multi-instance.md for the porting plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
