#!/usr/bin/env python3
"""Reference statistics for test_samudra_real: run the traced Samudra model on
its initial condition from Python, independently of the C++ code.

    samudra_reference.py IC_FILE MODEL_FILE

Prints min, max and unweighted mean over mask_2d for seven output channels.
"""
import sys, numpy as np, torch, netCDF4
ic_path, model_path = sys.argv[1], sys.argv[2]
forcing = ["TAUX","TAUY","surface_precipitation_rate","frozen_precipitation_rate","FLUS","FSUS","FLDS","FSDS","LHFLX","SHFLX"]
outputs = ["sst","ssh"] + [f"{s}{k}" for s in ["salinityCoarsened_","temperatureCoarsened_","velocityZonalCoarsened_","velocityMeridionalCoarsened_"] for k in range(19)] + ["ocean_sea_ice_fraction","iceVolumeTotal"]
inputs = ["LANDFRAC","sea_surface_fraction"] + forcing + outputs + forcing
ds = netCDF4.Dataset(ic_path)
x = np.stack([np.asarray(ds[n][:], dtype=np.float32) for n in inputs])[None]
mask = np.asarray(ds["mask_2d"][:]) == 1
m = torch.jit.load(model_path, map_location="cuda").eval()
torch._C._set_graph_executor_optimize(False)
with torch.no_grad():
    y = m(torch.from_numpy(x).cuda()).cpu().numpy()[0]
for name in ["sst","ssh","salinityCoarsened_0","temperatureCoarsened_0","velocityZonalCoarsened_0","ocean_sea_ice_fraction","iceVolumeTotal"]:
    v = y[outputs.index(name)][mask]
    print(f"{name:26s} min {v.min():8.2f} max {v.max():8.2f} mean {v.mean():7.2f}")
