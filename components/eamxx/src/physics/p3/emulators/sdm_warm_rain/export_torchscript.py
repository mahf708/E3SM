#!/usr/bin/env python3
"""Write the SDM warm-rain emulator as TorchScript, for the libtorch backend.

  export_torchscript.py warm_rain_emulator.pt sdm_warm_rain_torchscript.pt

Use the libtorch backend with option dtype: float64 (the network itself runs in
float32, as with the python backend).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdm_warm_rain import SdmWarmRain  # noqa: E402

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model_file', help='model file of the SDM_emulator bundle')
    p.add_argument('out_file', help='TorchScript file to write')
    args = p.parse_args()
    torch.jit.script(SdmWarmRain(args.model_file)).save(args.out_file)
    print('wrote', args.out_file)
