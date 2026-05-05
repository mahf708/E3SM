def compute_coll_SDM(numbin, dt, dmdlnr_bin, dndlnr_bin, m_to_n_factors):
    import numpy as np
    import torch
    from rom_utils import AESINDy, simulate

    # Constants for setup
    LATENT_DIM = 3
    POLY_ORDER = 2
    NBIN = numbin
    WEIGHT_DIR = "/global/cfs/cdirs/mp193/llin/github_code/cpp_SCREAM_python_ROM_interface_2026may04/ftorch_weights/erfAll_aesindy_weights.pth"
    M_SCALE = 0.01589495           # units: kg m^-3
    # ZLIM = np.load("./zlim_data.npz")['zlim']
    # ZLIM[-1, 0] = 0.0
    # ZLIM[-1, 1] = np.inf

    # set up the model: should be done only once and not need to be updated
    # on sequential calls to computation
    model = AESINDy(
        n_channels=1,
        n_bins=NBIN,
        n_latent=LATENT_DIM,
        poly_order=POLY_ORDER,
    )
    model.load_state_dict(torch.load(WEIGHT_DIR, weights_only=True))
    
    # normalize and scale data
    M_dlnr = np.sum(dmdlnr_bin)     # units: kg m^-3 (lnr)^-1
    x_scaled = dmdlnr_bin / M_dlnr  # units: (lnr)^-1
    M_scaled = M_dlnr / M_SCALE     # unitless

    # encode to latent space
    z_bf = np.zeros((LATENT_DIM + 1))
    z_bf[0:LATENT_DIM] = model.encoder(torch.Tensor(x_scaled)).detach().numpy()
    z_bf[LATENT_DIM] = M_scaled

    # step forward in time
    z_af = simulate(z_bf, [0, dt], model.dzdt)

    # decode; enforce mass conservation exactly, i.e. ignore dM/dt or latents[-1]
    x_scaled_af = model.decoder(torch.Tensor(z_af[-1, :-1])).detach().numpy()
    dmdlnr_af = x_scaled_af * M_dlnr

    # update dndlnr assuming constant piecewise distribution
    dndlnr_af = dmdlnr_af * m_to_n_factors

    return dmdlnr_af, dndlnr_af



