"""EAMxx python module for P3's warm-rain emulator (SDM-trained, ERF super-droplet LES).

Adapted from the SDM_emulator bundle (eamxx/warm_rain_emu.py). All model-specific numbers
(weights, normalization, architecture, gates, training envelope, number-rate constants)
come from the model file, so a retrained model is a file swap.

Called by P3Microphysics (eamxx_p3_warm_rain_emulator.cpp) as:
  init(model_file)
  check_timestep(dt)
  forward(qc, nc, qr, nr, rho, *outs)

forward() inputs : P3 dry mixing ratios qc, qr [kg/kg], nc, nr [#/kg] and dry density rho [kg/m3],
                   as (ncol, nlev) arrays, as P3 sees them at its warm-rain stage.
forward() outputs: written in place, per kg, GRID MEAN, in P3's sign conventions:
    qc2qr_autoconv_tend, qc2qr_accret_tend, ncautr, nc2nr_autoconv_tend, nc_accret_tend,
    nc_selfcollect_tend (<= 0, added to nc), nr_selfcollect_tend,
    use_cloud (1 where the six cloud-process rates should replace P3's),
    use_rain (1 where nr_selfcollect_tend should replace P3's)

The arrays are views of EAMxx fields. They are padded like P3's packs, so they are
not contiguous: they are read with np.asarray and written with out[...] = ...
"""
import os
import numpy as np
import torch
from torch import nn

SUPPORTED_FORMAT = 1
_ACT = dict(tanh=nn.Tanh, relu=nn.ReLU, silu=nn.SiLU, softplus=nn.Softplus)


class WarmRainEmulator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        a = cfg['architecture']; layers, n = [], a['n_in']
        for w in a['hidden']:
            layers += [nn.Linear(n, w), _ACT[a['activation']]()]; n = w
        layers += [nn.Linear(n, a['n_out']), _ACT[a['output_activation']]()]
        self.net = nn.Sequential(*layers)
        for name in ('x_mean', 'x_std', 'floors', 'y_scale', 'y_log_std'):
            self.register_buffer(name, torch.zeros(a['n_in'] if name in ('x_mean', 'x_std', 'floors') else a['n_out']))
        self.qc_gt, self.qr_gt = cfg['gates']['qc_gt'], cfg['gates']['qr_gt']

    def forward(self, x):   # x: (N, 4) per volume -> raw rates (N, 4) per volume, gated
        z = (torch.log10(torch.maximum(x, self.floors)) - self.x_mean) / self.x_std
        p = torch.expm1(self.net(z) * self.y_log_std) * self.y_scale
        cloud = x[:, 0] > self.qc_gt
        return p * torch.stack([cloud, cloud & (x[:, 2] > 0.0), cloud, x[:, 2] > self.qr_gt], 1).to(p.dtype)


model, cfg = None, None


def init(model_file):
    global model, cfg
    if not os.path.isfile(model_file):
        raise RuntimeError(f"p3_warm_rain_emulator: model file '{model_file}' not found")
    blob = torch.load(model_file, map_location=torch.device('cpu'), weights_only=True)
    cfg = blob['config']
    if cfg['format_version'] != SUPPORTED_FORMAT:
        raise RuntimeError(f"p3_warm_rain_emulator: model format {cfg['format_version']} not supported (expected {SUPPORTED_FORMAT})")
    names = [n for n, _ in cfg['contract']['inputs']]
    if names != ['qc', 'Nc', 'qr', 'Nr'] or cfg['transforms'] != dict(input='log10_floor_standardize', output='expm1_logstd_scale'):
        raise RuntimeError(f"p3_warm_rain_emulator: model contract {names} / {cfg['transforms']} does not match this module")
    model = WarmRainEmulator(cfg)
    model.load_state_dict(blob['state_dict'])
    model.eval()


def check_timestep(dt):
    """Warn if the physics step is far from the period the model's targets average over."""
    T = cfg['contract']['target_period_s']
    if not (0.5 * T <= dt <= 2.0 * T):
        print(f'p3_warm_rain_emulator WARNING: dt = {dt} s, but the model was trained on {T}-s mean rates', flush=True)


def masks(qc_v, nc_v, qr_v, nr_v):
    """Training-envelope masks from the model file: (cloud processes, rain self-collection)."""
    c, r = cfg['envelope']['cloud'], cfg['envelope']['rain']
    use_cloud = (qc_v > cfg['gates']['qc_gt']) & (qc_v <= c['qc_max']) & (nc_v >= c['nc_min']) & (nc_v <= c['nc_max']) \
        & (qr_v > c['qr_min']) & (qr_v <= c['qr_max']) & (nr_v <= c['nr_max'])
    use_rain = (qr_v <= r['qr_max']) & (nr_v <= r['nr_max'])
    return use_cloud, use_rain


def raw_rates(qc_v, nc_v, qr_v, nr_v):
    """Per-volume emulator rates (N, 4): AU, AC, SC_c, SC_r."""
    x = np.stack([qc_v, nc_v, qr_v, nr_v], -1)
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).cpu().numpy().astype(np.float64)


def p3_tendencies(qc, nc, qr, nr, rho):
    """Per-kg P3-ready tendencies and masks for flat arrays."""
    qc_v, nc_v, qr_v, nr_v = qc * rho, nc * rho, qr * rho, nr * rho
    au, ac, scc, scr = (raw_rates(qc_v, nc_v, qr_v, nr_v) / rho[:, None]).T
    n = cfg['number_rates']
    m_star = 4.0 / 3.0 * np.pi * 1000.0 * n['embryo_radius_m'] ** 3
    nc_over_qc = np.where(qc > 0, nc / np.where(qc > 0, qc, 1.0), 0.0)
    use_cloud, use_rain = masks(qc_v, nc_v, qr_v, nr_v)
    return dict(qc2qr_autoconv_tend=au, qc2qr_accret_tend=ac, ncautr=au / m_star,
                nc2nr_autoconv_tend=n['cloud_drops_per_embryo'] * au / m_star,
                nc_accret_tend=n['ac_n_factor'] * ac * nc_over_qc,
                nc_selfcollect_tend=-scc, nr_selfcollect_tend=scr,
                use_cloud=use_cloud.astype(np.float64), use_rain=use_rain.astype(np.float64))


OUTPUT_ORDER = ('qc2qr_autoconv_tend', 'qc2qr_accret_tend', 'ncautr', 'nc2nr_autoconv_tend', 'nc_accret_tend',
                'nc_selfcollect_tend', 'nr_selfcollect_tend', 'use_cloud', 'use_rain')


def forward(qc, nc, qr, nr, rho, *outs):
    """outs: nine arrays in OUTPUT_ORDER, updated in place."""
    if len(outs) != len(OUTPUT_ORDER):
        raise RuntimeError(f'p3_warm_rain_emulator.forward expects {len(OUTPUT_ORDER)} output arrays: {OUTPUT_ORDER}')
    flat = lambda v: np.asarray(v, dtype=np.float64).ravel()
    t = p3_tendencies(flat(qc), flat(nc), flat(qr), flat(nr), flat(rho))
    for out, name in zip(outs, OUTPUT_ORDER):
        out[...] = t[name].reshape(np.shape(out))
