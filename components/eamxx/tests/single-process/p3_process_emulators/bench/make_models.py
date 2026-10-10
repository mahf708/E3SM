"""TorchScript benchmark models: a pointwise MLP over (ncol, nlev) inputs.

  make_models.py OUTDIR    (writes whole_p3.pt, warm_rates.pt, rain_sed.pt)

Each output is base + 0 * mlp, so the model costs a full inference but returns
either zeros (rates, tendencies) or its first inputs unchanged (whole-process
state), keeping the physics sane while timing the emulator.
"""
import sys
import torch


class PointwiseMLP(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, n_pass: int, width: int = 64):
        super().__init__()
        self.n_out = n_out
        self.n_pass = n_pass  # outputs that return inputs[0..n_pass) unchanged
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, n_out)).double()

    def forward(self, xs: list[torch.Tensor]):
        x = torch.stack(xs, dim=-1)              # (ncol, nlev, n_in)
        y = self.net(x) * 0.0                    # (ncol, nlev, n_out)
        outs: list[torch.Tensor] = []
        for o in range(self.n_out):
            base = xs[o] if o < self.n_pass else torch.zeros_like(xs[0])
            outs.append(base + y[..., o])
        return outs


def write(path, n_in, n_out, n_pass):
    # TorchScript needs a fixed arity and real source: write forward(x0, ..., xN) to a module
    import importlib, os
    m = PointwiseMLP(n_in, n_out, n_pass)
    m.requires_grad_(False)
    args = ", ".join(f"x{i}: torch.Tensor" for i in range(n_in))
    names = ", ".join(f"x{i}" for i in range(n_in))
    rets = ", ".join(f"o[{i}]" for i in range(n_out))
    mod = f"fixed{n_in}_{n_out}"
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), mod + ".py"), "w") as f:
        f.write(f"""import torch


class Fixed(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, {args}):
        o = self.inner([{names}])
        return ({rets},)
""")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    fixed = importlib.import_module(mod).Fixed(m)
    torch.jit.script(fixed).save(path)


if __name__ == "__main__":
    out = sys.argv[1]
    write(f"{out}/whole_p3.pt", 12, 10, 10)   # 10 state fields back, from 12 inputs
    write(f"{out}/warm_rates.pt", 5, 7, 0)    # 7 warm rates (zero) from qc, nc, qr, nr, rho
    write(f"{out}/rain_sed.pt", 4, 2, 0)      # qr, nr sedimentation tendencies (zero)
    print("ok")
