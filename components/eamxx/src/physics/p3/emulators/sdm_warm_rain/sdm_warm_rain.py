"""SDM-trained warm-rain emulator (ERF super-droplet LES), as a P3 process emulator.

The model file is the one from the SDM_emulator bundle (eamxx/warm_rain_emulator.pt):
weights, normalization, gates, training envelope and number-rate constants all
come from it, so a retrained model is a file swap.

The same torch module serves both inference backends of components/emulators:
  - python:   this module, with create_emulator(config) (model_path = the bundle file)
  - libtorch: the TorchScript file written by export_torchscript.py

Contract (see sdm_warm_rain.yaml):
  inputs : qc, nc, qr, nr [kg/kg, #/kg, dry], rho [kg/m3, dry], each (ncol, nlev)
  outputs: qc2qr_autoconv_tend, qc2qr_accret_tend, ncautr, nc2nr_autoconv_tend,
           nc_accret_tend, nc_selfcollect_tend, nr_selfcollect_tend   (cell averages,
           in P3's sign conventions), and the masks use_cloud, use_rain (0/1)
"""
import math
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

SUPPORTED_FORMAT = 1
OUTPUTS = ('qc2qr_autoconv_tend', 'qc2qr_accret_tend', 'ncautr', 'nc2nr_autoconv_tend',
           'nc_accret_tend', 'nc_selfcollect_tend', 'nr_selfcollect_tend', 'use_cloud', 'use_rain')
_ACT = dict(tanh=nn.Tanh, relu=nn.ReLU, silu=nn.SiLU, softplus=nn.Softplus)


class SdmWarmRain(nn.Module):
    def __init__(self, model_file: str):
        super().__init__()
        blob = torch.load(model_file, map_location='cpu', weights_only=True)
        cfg = blob['config']
        if cfg['format_version'] != SUPPORTED_FORMAT:
            raise RuntimeError(f"sdm_warm_rain: model format {cfg['format_version']} not supported")
        names = [n for n, _ in cfg['contract']['inputs']]
        if names != ['qc', 'Nc', 'qr', 'Nr'] or \
           cfg['transforms'] != dict(input='log10_floor_standardize', output='expm1_logstd_scale'):
            raise RuntimeError(f"sdm_warm_rain: model contract {names} / {cfg['transforms']} not supported")

        a = cfg['architecture']
        layers, n = [], a['n_in']
        for w in a['hidden']:
            layers += [nn.Linear(n, w), _ACT[a['activation']]()]
            n = w
        layers += [nn.Linear(n, a['n_out']), _ACT[a['output_activation']]()]
        self.net = nn.Sequential(*layers)
        for name in ('x_mean', 'x_std', 'floors', 'y_scale', 'y_log_std'):
            self.register_buffer(name, torch.zeros_like(blob['state_dict'][name]))
        self.load_state_dict(blob['state_dict'])
        self.requires_grad_(False)
        self.eval()

        g, c, r, nr = cfg['gates'], cfg['envelope']['cloud'], cfg['envelope']['rain'], cfg['number_rates']
        self.qc_gt, self.qr_gt = float(g['qc_gt']), float(g['qr_gt'])
        self.c_qc_max, self.c_nc_min, self.c_nc_max = float(c['qc_max']), float(c['nc_min']), float(c['nc_max'])
        self.c_qr_min, self.c_qr_max, self.c_nr_max = float(c['qr_min']), float(c['qr_max']), float(c['nr_max'])
        self.r_qr_max, self.r_nr_max = float(r['qr_max']), float(r['nr_max'])
        self.m_star = 4.0 / 3.0 * math.pi * 1000.0 * float(nr['embryo_radius_m']) ** 3
        self.drops_per_embryo = float(nr['cloud_drops_per_embryo'])
        self.ac_n_factor = float(nr['ac_n_factor'])
        self.target_period_s = float(cfg['contract']['target_period_s'])

    def forward(self, qc: torch.Tensor, nc: torch.Tensor, qr: torch.Tensor, nr: torch.Tensor,
                rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                            torch.Tensor]:
        shape = qc.shape
        qc, nc, qr, nr, rho = [v.reshape(-1).double() for v in (qc, nc, qr, nr, rho)]
        qc_v, nc_v, qr_v, nr_v = qc * rho, nc * rho, qr * rho, nr * rho

        # The network runs in float32, on per-volume quantities
        x = torch.stack([qc_v, nc_v, qr_v, nr_v], 1).float()
        z = (torch.log10(torch.maximum(x, self.floors)) - self.x_mean) / self.x_std
        p = torch.expm1(self.net(z) * self.y_log_std) * self.y_scale
        cloud = x[:, 0] > self.qc_gt
        gates = torch.stack([cloud, cloud & (x[:, 2] > 0.0), cloud, x[:, 2] > self.qr_gt], 1)
        raw = (p * gates.to(p.dtype)).double()

        au, ac, scc, scr = (raw / rho[:, None]).unbind(1)
        nc_over_qc = torch.where(qc > 0, nc / torch.where(qc > 0, qc, torch.ones_like(qc)), torch.zeros_like(qc))
        use_cloud = (qc_v > self.qc_gt) & (qc_v <= self.c_qc_max) & (nc_v >= self.c_nc_min) & \
                    (nc_v <= self.c_nc_max) & (qr_v > self.c_qr_min) & (qr_v <= self.c_qr_max) & \
                    (nr_v <= self.c_nr_max)
        use_rain = (qr_v <= self.r_qr_max) & (nr_v <= self.r_nr_max)

        out = (au, ac, au / self.m_star, self.drops_per_embryo * au / self.m_star,
               self.ac_n_factor * ac * nc_over_qc, -scc, scr, use_cloud.double(), use_rain.double())
        o0, o1, o2, o3, o4, o5, o6, o7, o8 = [v.reshape(shape) for v in out]
        return o0, o1, o2, o3, o4, o5, o6, o7, o8


class PythonBackendEmulator:
    """What the python backend of components/emulators calls: infer(inputs, outputs)."""

    def __init__(self, model_file: str):
        self.model = SdmWarmRain(model_file)

    def infer(self, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]):
        args = [torch.from_numpy(np.asarray(inputs[n])) for n in ('qc', 'nc', 'qr', 'nr', 'rho')]
        with torch.no_grad():
            res = self.model(*args)
        for name, v in zip(OUTPUTS, res):
            if name in outputs:
                outputs[name][...] = v.numpy()


def create_emulator(config):
    return PythonBackendEmulator(config['model_path'])
