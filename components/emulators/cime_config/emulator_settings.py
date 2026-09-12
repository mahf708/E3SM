"""Write an emulated component's `key: value` input file for CIME.

Each emulated component reads `<class>_in` (atm_in, ocn_in, ice_in): one
`key: value` per line.  buildnml computes the defaults from the case, and
`user_nl_<component>` lines in the same form override them.  A key the
component does not read is an error here, at case.setup, rather than a
setting silently ignored at run time.
"""

import os

from CIME.utils import expect, safe_copy


def read_user_settings(caseroot, compname, inst_string):
    """`key: value` lines from user_nl_<compname><inst_string>."""
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
            settings[key] = value
    return settings


def write_input_file(case, caseroot, compname, class_name, defaults, known):
    """Write <class>_in for every instance into Buildconf and RUNDIR."""
    rundir = case.get_value("RUNDIR")
    ninst = case.get_value("NINST_{}".format(class_name.upper())) or 1
    confdir = os.path.join(caseroot, "Buildconf", compname + "conf")
    os.makedirs(confdir, exist_ok=True)
    for inst in range(1, ninst + 1):
        inst_string = "_{:04d}".format(inst) if ninst > 1 else ""
        settings = dict(defaults)
        user = read_user_settings(caseroot, compname, inst_string)
        unknown = sorted(set(user) - set(known))
        expect(not unknown,
               "user_nl_{}{}: {} does not read {}; it reads {}".format(
                   compname, inst_string, compname, ", ".join(unknown),
                   ", ".join(sorted(known))))
        settings.update(user)
        filename = "{}_in{}".format(class_name, inst_string)
        path = os.path.join(confdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# {} settings, written by buildnml; change them in "
                    "user_nl_{}{}\n".format(compname, compname, inst_string))
            for key in sorted(settings):
                if settings[key] != "":
                    f.write("{}: {}\n".format(key, settings[key]))
        if os.path.isdir(rundir):
            safe_copy(path, os.path.join(rundir, filename))
