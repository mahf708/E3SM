"""Write the benchmark input yamls: P3 alone, stock and with emulators.

  make_inputs.py NCOL NLEV NSTEPS MODELS_DIR
"""
import sys

ncol, nlev, nsteps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
models = sys.argv[4]

WARM = "qc2qr_autoconv_tend, nc2nr_autoconv_tend, ncautr, nc_selfcollect_tend, qc2qr_accret_tend, nc_accret_tend, nr_selfcollect_tend"
STATE = "T_mid, qv, qc, nc, qr, nr, qi, ni, qm, bm"

cases = {
    "stock":           "",
    "warm_overwrite":  f"    process_emulators: [warm]\n    warm: {{backend: libtorch, model_path: {models}/warm_rates.pt, physics: run, inputs: [qc, nc, qr, nr, rho], outputs: [{WARM}], options: {{dtype: float64}}}}\n",
    "warm_replace":    f"    process_emulators: [warm]\n    warm: {{backend: libtorch, model_path: {models}/warm_rates.pt, physics: skip, inputs: [qc, nc, qr, nr, rho], outputs: [{WARM}], options: {{dtype: float64}}}}\n",
    "rain_sed_replace": f"    sedimentation_emulators: [rsed]\n    rsed: {{backend: libtorch, model_path: {models}/rain_sed.pt, physics: skip, inputs: [qr, nr, rho, dz], outputs: [qr_sed_tend, nr_sed_tend], options: {{dtype: float64}}}}\n",
    "whole_replace":   f"    field_emulators: [whole]\n    field_emulators_mode: replace\n    whole: {{backend: libtorch, model_path: {models}/whole_p3.pt, inputs: [{STATE}, p_mid, pseudo_density], outputs: [{STATE}], options: {{dtype: float64}}}}\n",
}
# The same cut points with the stub backend: the cost of the cut itself, without inference
for name in ["warm_overwrite", "warm_replace", "rain_sed_replace", "whole_replace"]:
    e = cases[name]
    for m in ["warm_rates.pt", "rain_sed.pt", "whole_p3.pt"]:
        e = e.replace(f"backend: libtorch, model_path: {models}/{m}", "backend: stub")
    cases[name + "_stub"] = e.replace(", options: {dtype: float64}", "")

for name, emus in cases.items():
    with open(f"input_{name}.yaml", "w") as f:
        f.write(f"""%YAML 1.1
---
time_stepping:
  time_step: 100
  run_t0: 2021-10-12-45000
  number_of_steps: {nsteps}

eamxx:
  atm_procs_list: [p3]
  p3:
    max_total_ni: 740.0e3
    do_prescribed_ccn: false
    use_hetfrz_classnuc: false
    process_emulators_limit_self_collection: false
{emus}
grids_manager:
  type: mesh_free
  grids_names: [physics]
  physics:
    type: point_grid
    number_of_global_columns:   {ncol}
    number_of_vertical_levels:  {nlev}

initial_conditions:
  T_mid: 268.0
  T_prev_micro_step: 268.0
  qv: 3.0e-3
  qv_prev_micro_step: 3.0e-3
  qc: 4.0e-4
  nc: 5.0e7
  qr: 2.0e-5
  nr: 1.0e4
  qi: 1.0e-4
  qm: 1.0e-5
  ni: 1.0e5
  bm: 2.5e-8
  p_mid: 8.0e4
  p_dry_mid: 7.96e4
  pseudo_density: 1000.0
  pseudo_density_dry: 996.0
  cldfrac_tot: 0.6
  nc_nuceat_tend: 0.0
  nccn: 0.0
  ni_activated: 0.0
  inv_qc_relvar: 1.0
  precip_liq_surf_mass: 0.0
  precip_ice_surf_mass: 0.0
  hetfrz_immersion_nucleation_tend: 0.0
  hetfrz_contact_nucleation_tend: 0.0
  hetfrz_deposition_nucleation_tend: 0.0
...
""")
print("\n".join(cases))
