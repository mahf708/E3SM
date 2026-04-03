def compute(qc, pseudo_density):
    """Liquid water path: vertical integral of cloud liquid."""
    integrand = qc * pseudo_density / 9.80616
    return col_sum(integrand)
