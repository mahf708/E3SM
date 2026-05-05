def find_nearest(array, value):
    import numpy as np
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def get_M_to_N_factors(numbin, rhow, r_edg):
    import numpy as np
    m_to_n_factors = np.ones((numbin)) / 4 / np.pi / rhow
    for i in range(numbin):
        m_to_n_factors[i] = m_to_n_factors[i] / np.log(r_edg[i+1] / r_edg[i]) 
        m_to_n_factors[i] = m_to_n_factors[i] * (1 / r_edg[i]**3 - 1 / r_edg[i+1]**3)
    return m_to_n_factors

def ROM_interface(qc_in, nc_in, qr_in, nr_in, muc_in, mur_in, qsmall, dt_in):
    import numpy as np
    import sys
    import py_ROM
    from scipy.integrate import quad

    """
    Description: A python interface that links to SDM python.
    Here reads in cloud variables from C++, creates assumed gamma size distributions based on the bulk properties, put into bins, and passes the bin distribution to SDM calculations, and retrieves the calculated rain tendency rate terms due to collision-coalescence and passes them back to C++.

    input: 
    qc_in: cloud liquid water content, kg/m3
    nc_in: cloud liquid droplet number concentration, 1/m3
    qr_in: rain water content, kg/m3
    nr_in: raindrop number concentration, 1/m3
    muc_in: spectral width of assumed gamma size distribution function for cloud liquid water, 
    mur_in: spectral width of assumed gamma size distribution function for rain,
    qsmall: low bound threshold.
    dt_in: host model time step.

    output:
    qc_tend_out: time rate of change of cloud liquid water due to collision-coalescence for a given time step, kg/m3/s
    nc_tend_out: time rate of change of cloud liquid droplet number due to collision-coalescence for a given time step, 1/m3/s
    qr_tend_out: time rate of change of rain due to collision-coalescence for a given time step, kg/m3/s
    nr_tend_out: time rate of change of raindrop number due to collision-coalescence for a given time step, 1/m3/s
    """
    # CONSTANTS
    # radius bins
    NUMBIN = 64
    NUMEDG = NUMBIN + 1
    EDGR = 5e-3                    # unit: m
    EDGL = 1e-6                    # unit: m
    R_EDG = np.exp(np.linspace(np.log(EDGL), np.log(EDGR), NUMEDG))
    DLNR = (np.log(EDGR) - np.log(EDGL)) / NUMEDG

    RHOW = 1000.0   # liquid water density, unit: kg/m3
    M_TO_N_FACTORS = get_M_to_N_factors(NUMBIN, RHOW, R_EDG)
    ### determine cloud liquid and rain cutoff size
    ### for simplicity, use a fixed radius threshold of 40 um to differentiate raindrops from cloud droplets for now (reference to 40 um: Gettelman et al 2021 JAMES; Geoffroy et al., 2014 ACP; Azimi et al. 2024 JAMES: 50um radius)
    val,cutoff_idx = find_nearest(R_EDG, 50 * 1.e-6)
    #print('cutoff: ', val, ' ', cutoff_idx)

    ### a scalar
    Nc = nc_in
    Qc = qc_in
    muc= muc_in

    Nr = nr_in
    Qr = qr_in
    mur= mur_in
    
    ### convert bulk aggregated properties to bin necessary for SDM calculations
    ### assumed gamma size distribution
    lambdac = 0.0
    N0c     = 0.0
    lambdar = 0.0
    N0r     = 0.0
    dndlnr_bin     = np.zeros((NUMBIN))
    dmdlnr_bin     = np.zeros((NUMBIN))

    if qc_in > qsmall:
        (N0c, lambdac) = get_gamma_params(Nc, Qc, muc, RHOW=RHOW)
        #print('cloud liquid N0, lambda= ', N0c, lambdac)
        for i in range(NUMBIN):
            n_cloud, _ = quad(lambda D: gamma_distribution(D, N0c, lambdac, muc),
                        R_EDG[i]*2, R_EDG[i+1]*2)
            m_cloud, _ = quad(lambda D: np.pi/6 * RHOW * D**3 * 
                                    gamma_distribution(D, N0c, lambdac, muc),
                                    R_EDG[i]*2, R_EDG[i+1]*2)
            dndlnr_bin[i] += n_cloud / DLNR
            dmdlnr_bin[i] += m_cloud / DLNR

    if qr_in > qsmall:
        (N0r, lambdar) = get_gamma_params(Nr, Qr, mur)
        #print('rain N0r, lambda= ', N0r, lambdar)
        for i in range(NUMBIN):
            n_rain, _ = quad(lambda D: gamma_distribution(D, N0r, lambdar, mur),
                        R_EDG[i]*2, R_EDG[i+1]*2)
            m_rain, _ = quad(lambda D: np.pi/6 * RHOW * D**3 * 
                                    gamma_distribution(D, N0r, lambdar, mur),
                                    R_EDG[i]*2, R_EDG[i+1]*2)
            dndlnr_bin[i] += n_rain / DLNR
            dmdlnr_bin[i] += m_rain / DLNR

    if (qc_in > qsmall) or (qr_in > qsmall):        
        #####################################################################
        ### call SDM emulator function, pass in "initial" PSD, return new PSD after the collision-coalescence processes with a timte step = 100 sec
        dt = dt_in ##100.0                                             # unit: sec
        new_dmdlnr_bin, new_dndlnr_bin = py_ROM.compute_coll_SDM(NUMBIN, dt, dmdlnr_bin[:], dndlnr_bin[:], M_TO_N_FACTORS)
        #####################################################################
    
        ### derive tendencies
        ### cloud liquid 
        cld_dsd_nbf  = np.sum(dndlnr_bin[0:cutoff_idx+1]) * DLNR
        cld_dsd_mbf  = np.sum(dmdlnr_bin[0:cutoff_idx+1]) * DLNR
        
        cld_dsd_naf  = np.sum(new_dndlnr_bin[0:cutoff_idx+1]) * DLNR
        cld_dsd_maf  = np.sum(new_dmdlnr_bin[0:cutoff_idx+1]) * DLNR
        
        ### rain
        rain_dsd_nbf = np.sum(dndlnr_bin[cutoff_idx+1:NUMBIN]) * DLNR
        rain_dsd_mbf = np.sum(dmdlnr_bin[cutoff_idx+1:NUMBIN]) * DLNR
    
        rain_dsd_naf = np.sum(new_dndlnr_bin[cutoff_idx+1:NUMBIN]) * DLNR
        rain_dsd_maf = np.sum(new_dmdlnr_bin[cutoff_idx+1:NUMBIN]) * DLNR
    
        ### total liquid for mass conservation test
        qliqtotbf = np.sum(dmdlnr_bin)
    
        qliqtotaft= np.sum(new_dmdlnr_bin)

        mass_noncons = (qliqtotaft - qliqtotbf)/qliqtotbf * 100.0

        if (mass_noncons < 1e-4):
            #print(f'Mass conservation verified to {mass_noncons:.8f}% error')
            pass
        else:
            raise RuntimeError(f'Mass is not conserved. Check! {mass_noncons:.5f}% error')

        # Tendencies: qc, nc must be decreasing; qr must be increasing
        ## LL Note:
        if (cld_dsd_maf - cld_dsd_mbf) >= 0.0:
            qc_tend_out = 0.0
            qr_tend_out = 0.0
            nc_tend_out = 0.0
            nr_tend_out = 0.0

        else:
        ## LL Note
            qc_tend_out = np.minimum((cld_dsd_maf - cld_dsd_mbf), 0)/dt   # unit: kg m-3 s-1
            nc_tend_out = np.minimum((cld_dsd_naf - cld_dsd_nbf), 0)/dt   # unit: m-3 s-1
            qr_tend_out = -qc_tend_out        # Enforce mass conservation exactly in bulk quantities
            nr_tend_out = (rain_dsd_naf - rain_dsd_nbf)/dt # unit: m-3 s-1

        if (qc_in < qsmall) and (qr_in > qsmall):
            qc_tend_out = 0.0
            qr_tend_out = 0.0
            
            nr_tend_out += nc_tend_out
            nc_tend_out = 0.0
    
    ### no cloud, skip all
    else:
        qc_tend_out = 0.0
        nc_tend_out = 0.0
        qr_tend_out = 0.0
        nr_tend_out = 0.0
        
    return qc_tend_out, nc_tend_out, qr_tend_out, nr_tend_out # Note that these are really rho_Q and rho_N tendencies (rho of air)

def gamma_distribution(D, N0, lambda0, mu):
    import numpy as np
    return N0 * D**mu * np.exp(-lambda0 * D)

def get_gamma_params(Nt, qt, mu, rho_air = 1.0, RHOW = 1000.0):
    import math
    import numpy as np
    gamma_ratio = math.gamma(mu + 4) / math.gamma(mu + 1)
    lambda0 = (np.pi / 6 * RHOW * Nt * gamma_ratio / (qt * rho_air))**(1.0/3.0)  # unit: meter -1
    N0      = Nt*lambda0**(mu+1)/math.gamma(mu+1)   # unit: m^(-mu-4)
    return (N0, lambda0)
