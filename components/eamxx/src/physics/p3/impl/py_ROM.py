def compute_coll_SDM(numbin, dt, dmdlnr_bin, dndlnr_bin, m_to_n_factors):
    import numpy as np
    import torch
    from rom_utils import AESINDy, simulate

    # Constants for setup
    LATENT_DIM = 3
    POLY_ORDER = 2
    NBIN = numbin
    # Path to local artifact bundle (weights + latent-limit data) on Perlmutter.
    # See cpp_SCREAM_python_ROM_interface for the source of truth.
    ARTIFACT_DIR = "/pscratch/sd/m/mahf708/e3sm-repo/cpp_SCREAM_python_ROM_interface"
    WEIGHT_DIR = f"{ARTIFACT_DIR}/aesindy_weights_pysdm.pth"
    M_SCALE = 0.01046717958952477  # units: kg m^-3 (lnr)^-1 (matches aesindy_weights_pysdm.pth)
    ZLIM = np.load(f"{ARTIFACT_DIR}/zlim_data.npz")['zlim']
    # Override mass-channel limits so the bulk-mass latent is unclipped.
    ZLIM[-1, 0] = 0.0
    ZLIM[-1, 1] = np.inf

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
    z_af = simulate(z_bf, [0, dt], model.dzdt, ZLIM)

    # decode; enforce mass conservation exactly, i.e. ignore dM/dt or latents[-1]
    x_scaled_af = model.decoder(torch.Tensor(z_af[-1, :-1])).detach().numpy()
    dmdlnr_af = x_scaled_af * M_dlnr

    # update dndlnr assuming constant piecewise distribution
    dndlnr_af = dmdlnr_af * m_to_n_factors

    return dmdlnr_af, dndlnr_af



