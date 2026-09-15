"""Check deterministic restart equivalence and optional backend agreement."""
import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('backends', nargs='+')
a = p.parse_args()
reference = None
for backend in a.backends:
    def read(mode):
        return np.loadtxt(a.directory / f'{backend}_{mode}.csv', delimiter=',', skiprows=1)
    full, segment, resume = (read(m) for m in ('full', 'segment', 'resume'))
    assert full.shape[0] == 49 and segment.shape[0] == 8 and resume.shape[0] == 42
    assert np.isfinite(full).all()
    np.testing.assert_array_equal(full, np.vstack((segment, resume[1:])))
    np.testing.assert_array_equal(segment[-1], resume[0])
    if reference is not None:
        np.testing.assert_allclose(full, reference, rtol=2e-6, atol=2e-6)
    reference = full
    print(backend, 'restart and backend comparisons passed')
