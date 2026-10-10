"""Numpy implementation of the P3 warm-rain emulator python contract, for tests.

It evaluates the same text model file as the device-native backend
(src/physics/p3/warm_rain_emulator/p3_warm_rain_mlp.hpp), so the python and
kokkos backends of P3Microphysics can be compared without torch. See
src/physics/p3/warm_rain_emulator/p3_warm_rain_emulator.py for the contract.
"""
import numpy as np

_ACT = dict(tanh=np.tanh,
            relu=lambda x: np.maximum(x, 0),
            silu=lambda x: x / (1 + np.exp(-x)),
            softplus=lambda x: np.maximum(x, 0) + np.log(1 + np.exp(-np.abs(x))))

model = None


def init(model_file):
    global model
    tok = open(model_file).read().split()
    pos = 0

    def take(n=1, conv=float):
        nonlocal pos
        v = [conv(t) for t in tok[pos:pos + n]]
        pos += n
        return v

    def expect(key):
        if take(1, str)[0] != key:
            raise RuntimeError(f'p3_warm_rain_mlp_numpy: expected {key} in {model_file}')

    m = {}
    expect('p3_warm_rain_mlp'); take(1, int)
    expect('activation'); m['act'], m['out_act'] = take(2, str)
    expect('widths'); nl = take(1, int)[0]; w = take(nl + 1, int)
    for key, n in (('x_mean', 4), ('x_std', 4), ('floors', 4), ('y_scale', 4), ('y_log_std', 4),
                   ('gates', 2), ('envelope_cloud', 6), ('envelope_rain', 2), ('number_rates', 3)):
        expect(key); m[key] = np.array(take(n))
    m['layers'] = []
    for l in range(nl):
        expect('weight'); W = np.array(take(w[l + 1] * w[l])).reshape(w[l + 1], w[l])
        expect('bias'); b = np.array(take(w[l + 1]))
        m['layers'].append((W, b))
    expect('end')
    model = m


def check_timestep(dt):
    pass


def forward(qc, nc, qr, nr, rho, *outs):
    m = model
    flat = lambda v: np.asarray(v, dtype=np.float64).ravel()
    qc, nc, qr, nr, rho = flat(qc), flat(nc), flat(qr), flat(nr), flat(rho)
    x = np.stack([qc * rho, nc * rho, qr * rho, nr * rho])          # (4, N), per volume

    a = (np.log10(np.maximum(x, m['floors'][:, None])) - m['x_mean'][:, None]) / m['x_std'][:, None]
    for l, (W, b) in enumerate(m['layers']):
        act = m['out_act'] if l == len(m['layers']) - 1 else m['act']
        a = _ACT[act](W @ a + b[:, None])
    y = np.expm1(a * m['y_log_std'][:, None]) * m['y_scale'][:, None]
    qc_gt, qr_gt = m['gates']
    cloud = x[0] > qc_gt
    y *= np.stack([cloud, cloud & (x[2] > 0), cloud, x[2] > qr_gt])

    au, ac, scc, scr = y / rho
    r_emb, drops_per_embryo, ac_n_factor = m['number_rates']
    m_star = 4.0 / 3.0 * np.pi * 1000.0 * r_emb ** 3
    nc_over_qc = np.where(qc > 0, nc / np.where(qc > 0, qc, 1.0), 0.0)
    c, r = m['envelope_cloud'], m['envelope_rain']
    use_cloud = (x[0] > qc_gt) & (x[0] <= c[0]) & (x[1] >= c[1]) & (x[1] <= c[2]) \
        & (x[2] > c[3]) & (x[2] <= c[4]) & (x[3] <= c[5])
    use_rain = (x[2] <= r[0]) & (x[3] <= r[1])

    res = (au, ac, au / m_star, drops_per_embryo * au / m_star, ac_n_factor * ac * nc_over_qc,
           -scc, scr, use_cloud.astype(np.float64), use_rain.astype(np.float64))
    for out, v in zip(outs, res):
        out[...] = v.reshape(np.shape(out))
