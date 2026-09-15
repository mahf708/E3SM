"""Generate a small software demonstration with the ACE2 channel contract."""
import argparse
from pathlib import Path
import numpy as np
from scipy.io import netcdf_file
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--torch', action='store_true', help='also export a TorchScript fixture')
    args = parser.parse_args()
    dest = args.directory.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    ny, nx = 2, 4
    with netcdf_file(dest / 'grid.nc', 'w') as f:
        f.createDimension('grid_size', nx * ny)
        f.createDimension('grid_rank', 2)
        f.createVariable('grid_dims', 'i', ('grid_rank',))[:] = [nx, ny]
        for name, values, units in (
            ('grid_center_lat', np.repeat([-45., 45.], nx), b'degrees_north'),
            ('grid_center_lon', np.tile([0., 90., 180., 270.], ny), b'degrees_east'),
            ('grid_area', np.full(nx * ny, 4 * np.pi / (nx * ny)), b'radians^2')):
            v = f.createVariable(name, 'd', ('grid_size',)); v[:] = values; v.units = units
        f.createVariable('grid_imask', 'i', ('grid_size',))[:] = 1
    values = {'LANDFRAC': 0., 'OCNFRAC': 1., 'ICEFRAC': 0., 'PHIS': 0.,
              'PS': 100000., 'TS': 285., 'Sf_lfrac': 0., 'Sf_ofrac': 1.,
              'Sf_ifrac': 0., 'Sx_t': 285.}
    for stem, value in [('T', 280.), ('specific_total_water', 0.005), ('U', 5.), ('V', 2.)]:
        values.update({f'{stem}_{k}': value for k in range(8)})
    with netcdf_file(dest / 'ic.nc', 'w') as f:
        f.createDimension('lat', ny); f.createDimension('lon', nx)
        for name, value in values.items():
            f.createVariable(name, 'd', ('lat', 'lon'))[:] = np.full((ny, nx), value)
    backends = {'python_fixture': {'backend': 'python', 'model_path': '',
                'python_module': 'atmosphere_fixture',
                'python_path': str(root / 'common/tests/fixtures')}}
    if args.torch:
        import torch
        class Model(torch.nn.Module):
            def forward(self, x):
                y = x[:, 5:39].clone()
                y[:, 0] = y[:, 0] + 10.
                y[:, 2:10] = y[:, 2:10] + 0.125
                flux = torch.tensor([80., 20., 1e-5, 350., 240., 300., 150., 30., 100., 0.], device=x.device)
                return torch.cat((y, flux[None, :, None, None].expand(x.size(0), 10, x.size(2), x.size(3))), 1)
        torch.jit.trace(Model().eval(), torch.zeros(1, 39, ny, nx)).save(str(dest / 'model.pt'))
        backends.update(libtorch={'backend': 'libtorch', 'model_path': 'model.pt', 'device': 'cpu'},
                        python_torch={'backend': 'python', 'model_path': 'model.pt', 'device': 'cpu',
                                      'python_module': 'e3sm_emulator.torchscript'})
    for backend, inference in backends.items():
        component = {'spec': str(root / 'specs/ace2-eamv3.yaml'), 'coupler_dt': 1800,
                     'grid': {'file': 'grid.nc', 'domain': 'full'},
                     'initial_condition': 'ic.nc', 'inference': inference}
        (dest / f'{backend}_atm.yaml').write_text(yaml.safe_dump(component))
        for mode, steps, tod in [('full', 48, 0), ('segment', 7, 0), ('resume', 41, 12600)]:
            run = {'component': f'{backend}_atm.yaml', 'start_ymd': 19710101, 'start_tod': tod,
                   'steps': steps, 'surface': {'file': 'ic.nc', 'variables': {n: n for n in ('Sf_lfrac', 'Sf_ofrac', 'Sf_ifrac', 'Sx_t')}},
                   'output': f'{backend}_{mode}.csv'}
            if mode == 'segment': run['restart_out'] = f'{backend}.restart'
            if mode == 'resume': run['restart_in'] = f'{backend}.restart'
            (dest / f'{backend}_{mode}.yaml').write_text(yaml.safe_dump(run))
    print(dest)


if __name__ == '__main__':
    main()
