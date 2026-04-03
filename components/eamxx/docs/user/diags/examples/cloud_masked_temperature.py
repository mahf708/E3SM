def compute(T_mid, cldfrac_liq):
    """Temperature masked to cloudy columns only."""
    mask = gt(cldfrac_liq, 0.5)
    return where(mask, T_mid, -9999.0)
