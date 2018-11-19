import time
from pathlib import Path

import numpy as np
import contextlib


from sync_numpy import ShadowedNumpyMemmap


def test_file_removal():
    m = np.eye(8, dtype='float32')
    with contextlib.closing(ShadowedNumpyMemmap(m)) as mm:
        mm[:,:]
        filename = Path(mm.filename)
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


def test_overlapping_chunks():
    m = np.eye(8, dtype='float32')
    with contextlib.closing(ShadowedNumpyMemmap(m)) as mm:
        assert mm.copy_ratio == 0
        mm[1:]
        assert mm.copy_ratio > 0
        mm[:5]
        assert mm.copy_ratio == 1


def test_background_syncing():
    m = np.eye(8, dtype='float32')
    with contextlib.closing(ShadowedNumpyMemmap(m)) as mm:
        time.sleep(0.5)
        assert mm.copy_ratio == 0
        assert mm.sync_item_duration is None

        mm.bandwidth_share = 0.5

        starttime = time.time()
        while time.time() - starttime < 2:
            if mm.fully_copied:
                break
            time.sleep(.1)
        assert mm.copy_ratio == 1
        assert mm.fully_copied
        assert mm.sync_item_duration > 0

        print("mm.sync_item_duration =", mm.sync_item_duration)


def test_numpy_list_accessors():
    m = np.array(range(100), dtype='float32')
    with contextlib.closing(ShadowedNumpyMemmap(m)) as mm:
        i = [i in (1, 5, 99) for i in range(100)]
        assert (m[i] == mm[i]).all()

    with contextlib.closing(ShadowedNumpyMemmap(m)) as mm:
        i = [1, 4, 5, 6, 1, -99, -1, -4, -4]
        assert (m[i] == mm[i]).all()

    with contextlib.closing(ShadowedNumpyMemmap(m)) as mm:
        i = np.array([1, 4, 5, 6, 1, -99, -1, -4, -4])
        assert (m[i] == mm[i]).all()

    with contextlib.closing(ShadowedNumpyMemmap(m)) as mm:
        i = np.array([i in (1, 5, 99) for i in range(100)], dtype=np.bool8)
        assert (m[i] == mm[i]).all()
