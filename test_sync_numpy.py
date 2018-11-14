from pathlib import Path

import numpy as np
import contextlib

from sync_numpy import ShadowedNumpyMemmap


def test_equal():
    m = np.eye(8, dtype='float32')
    with contextlib.closing(ShadowedNumpyMemmap(m)) as mm:
        filename = Path(mm.filename)
        assert filename.exists()
    assert not filename.exists()


def test_equal():
    m = np.eye(8, dtype='float32')

    with contextlib.closing(ShadowedNumpyMemmap(m)) as mm:
        assert mm.copy_ratio == 0

        assert mm.ndim == m.ndim
        assert mm.shape == m.shape
        assert mm.dtype == m.dtype
        assert len(mm) == len(m)
        assert (m == mm[:]).all()
        assert (m[1] == mm[1]).all()

        assert mm.copy_ratio == 1
        assert mm.fully_copied
