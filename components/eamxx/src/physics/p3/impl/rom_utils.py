import torch
from torch import nn
from torch.nn import (
    Linear,
    ReLU,
    Sigmoid,
    Identity,
    Softmax,
)
from scipy.special import binom

from itertools import combinations_with_replacement

"""
Classes and functionality are part of a larger codebase and study by de Jong et al.
on reduced-order modeling for droplet coalescence.
"""

class AESINDy(torch.nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_bins=64,
        n_latent=3,
        poly_order=2,
        sequential_thresholding=False,
    ):
        super(AESINDy, self).__init__()
        self.poly_order = poly_order

        assert n_channels == 1
        self.encoder = FFNNEncoder(n_bins=n_bins, n_latent=n_latent)
        self.decoder = FFNNDecoder(
            n_bins=n_bins, n_latent=n_latent, distribution=True
        )
        self.dzdt = SINDyDeriv(
            n_latent=n_latent + 1,
            poly_order=poly_order,
            use_thresholds=sequential_thresholding,
        )

    def forward(self, bin0, M):
        z0 = self.encoder(bin0)
        dzMdt = self.dzdt(z0, M)
        dzdt = dzMdt[:, :, :-1]
        return dzdt
    
class FFNNEncoder(torch.nn.Module):
    def __init__(self, n_bins=64, n_latent=3):
        super(FFNNEncoder, self).__init__()
        self.n_bins = n_bins
        self.layer1 = Linear(n_bins, int(n_bins / 2))
        self.activation1 = ReLU()
        self.layer2 = Linear(int(n_bins / 2), int(n_bins / 4))
        self.activation2 = ReLU()
        self.layer3 = Linear(int(n_bins / 4), int(n_bins / 8))
        self.activation3 = ReLU()
        self.layer4 = Linear(int(n_bins / 8), n_latent)
        self.activation4 = Identity()

        self.apply(self.init_weights)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.act = [
            self.activation1,
            self.activation2,
            self.activation3,
            self.activation4,
        ]

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)

        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def get_weights(self):
        weights = []
        biases = []
        for i, layer in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)

    def set_weights(self, weights, biases):
        for i, layer in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]


class FFNNDecoder(torch.nn.Module):
    def __init__(self, n_bins=64, n_latent=3, distribution=True):
        super(FFNNDecoder, self).__init__()

        self.n_bins = n_bins
        self.layer1 = Linear(n_latent, int(n_bins / 8))
        self.layer2 = Linear(int(n_bins / 8), int(n_bins / 4))
        self.layer3 = Linear(int(n_bins / 4), int(n_bins / 2))
        self.layer4 = Linear(int(n_bins / 2), n_bins)
        self.activation1 = ReLU()
        self.activation2 = ReLU()
        self.activation3 = ReLU()
        if distribution:
            self.activation4 = Softmax(dim=-1)
        else:
            self.activation4 = Sigmoid()

        self.apply(self.init_weights)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.act = [
            self.activation1,
            self.activation2,
            self.activation3,
            self.activation4,
        ]

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)

        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def get_weights(self):
        weights = []
        biases = []
        for i, layer in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)

    def set_weights(self, weights, biases):
        for i, layer in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]

class SINDyDeriv(torch.nn.Module):
    def __init__(self, n_latent=10, poly_order=2, use_thresholds=False):
        super(SINDyDeriv, self).__init__()
        self.library_size = library_size(n_latent, poly_order)
        self.n_latent = n_latent
        self.poly_order = poly_order

        self.sindy_coeffs = torch.nn.Linear(
            self.library_size, self.n_latent, bias=False
        )
        self.use_thresholds = use_thresholds
        if use_thresholds:
            self.mask = torch.ones_like(self.sindy_coeffs.weight.data, dtype=bool)

        self.apply(self.init_weights)

    def forward(self, z, M=None):
        if M is not None:
            latent = torch.cat([z, M], dim=-1)
        else:
            latent = z
        library = sindy_library_tensor(latent, self.n_latent, self.poly_order)
        if self.use_thresholds:
            self.sindy_coeffs.weight.data = self.sindy_coeffs.weight.data * self.mask
        dldt = self.sindy_coeffs(library)
        return dldt

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def get_coeffs(self):
        return self.sindy_coeffs.weight.data * self.mask

    def update_mask(self, new_mask):
        self.mask = self.mask * new_mask
        self.sindy_coeffs.weight.data = self.mask * self.sindy_coeffs.weight.data

def library_size(n, poly_order):
    from scipy.special import binom
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    return l

def sindy_library_tensor(z, latent_dim, poly_order):
    from itertools import combinations_with_replacement
    import torch
    library_dim = library_size(latent_dim, poly_order)
    if len(z.shape) == 1:
        z = z.unsqueeze(0)
    if len(z.shape) == 2:
        z = z.unsqueeze(1)
    new_library = torch.zeros(z.shape[0], z.shape[1], library_dim)

    # i = 0: constant
    idx = 0
    new_library[:, :, idx] = 1.0

    idx += 1
    # i = 1:nl + 1 -> first order
    if poly_order >= 1:
        new_library[:, :, idx : idx + latent_dim] = z

    idx += latent_dim
    # second order
    if poly_order >= 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                new_library[:, :, idx] = z[:, :, i] * z[:, :, j]
                idx += 1

    # third order+
    for order in range(3, poly_order + 1):
        for idxs in combinations_with_replacement(range(latent_dim), order):
            term = z[:, :, idxs[0]]
            for i in idxs[1:]:
                term = term * z[:, :, i]
            new_library[:, :, idx] = term
            idx += 1

    return new_library

def simulate(z0, T, dz_network):
    from scipy.integrate import solve_ivp
    import torch
    def f(t, z):
        n_latent = z.size
        dz = dz_network(torch.Tensor(z)).squeeze().detach().numpy()
        return dz

    sol = solve_ivp(f, [T[0], T[-1]], z0, method="RK45", t_eval=T)
    Z = sol.y.T
    return Z