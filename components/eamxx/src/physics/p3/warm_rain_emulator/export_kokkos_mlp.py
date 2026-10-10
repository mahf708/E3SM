#!/usr/bin/env python3
"""Write a warm-rain emulator model file (.pt, as read by p3_warm_rain_emulator.py) as the
text file read by the device-native backend (p3_warm_rain_mlp.hpp).

  python export_kokkos_mlp.py warm_rain_emulator.pt warm_rain_emulator.txt
"""
import argparse
import torch


def export(model_file, out_file):
    blob = torch.load(model_file, map_location='cpu', weights_only=True)
    cfg, sd = blob['config'], blob['state_dict']
    if cfg['format_version'] != 1 or cfg['transforms'] != dict(input='log10_floor_standardize', output='expm1_logstd_scale'):
        raise RuntimeError(f'{model_file}: unsupported model format or transforms')
    a = cfg['architecture']
    widths = [a['n_in']] + list(a['hidden']) + [a['n_out']]
    layers = sorted({int(k.split('.')[1]) for k in sd if k.startswith('net.') and k.endswith('.weight')})
    if len(layers) != len(widths) - 1:
        raise RuntimeError(f'{model_file}: state dict does not match the architecture')

    fmt = lambda v: ' '.join(f'{float(x):.17g}' for x in torch.as_tensor(v).flatten())
    g, c, r, n = cfg['gates'], cfg['envelope']['cloud'], cfg['envelope']['rain'], cfg['number_rates']
    lines = [
        'p3_warm_rain_mlp 1',
        f"activation {a['activation']} {a['output_activation']}",
        f'widths {len(widths) - 1} ' + ' '.join(map(str, widths)),
        'x_mean ' + fmt(sd['x_mean']),
        'x_std ' + fmt(sd['x_std']),
        'floors ' + fmt(sd['floors']),
        'y_scale ' + fmt(sd['y_scale']),
        'y_log_std ' + fmt(sd['y_log_std']),
        f"gates {g['qc_gt']!r} {g['qr_gt']!r}",
        'envelope_cloud ' + ' '.join(repr(float(c[k])) for k in ('qc_max', 'nc_min', 'nc_max', 'qr_min', 'qr_max', 'nr_max')),
        'envelope_rain ' + ' '.join(repr(float(r[k])) for k in ('qr_max', 'nr_max')),
        'number_rates ' + ' '.join(repr(float(n[k])) for k in ('embryo_radius_m', 'cloud_drops_per_embryo', 'ac_n_factor')),
    ]
    for l in layers:
        lines.append('weight ' + fmt(sd[f'net.{l}.weight']))   # row-major (out, in)
        lines.append('bias ' + fmt(sd[f'net.{l}.bias']))
    lines.append('end')
    with open(out_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model_file')
    p.add_argument('out_file')
    args = p.parse_args()
    export(args.model_file, args.out_file)
