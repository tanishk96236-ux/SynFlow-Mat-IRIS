import os
import pathlib
import sys

# BEFORE numpy or torch is imported, or it has no effect.
#
# Without this, any numpy.linalg call aborts the process on the development
# machine -- two copies of the Intel OpenMP runtime get loaded, one through
# numpy's MKL and one through torch, and MKL kills the process rather than
# risk them fighting.
#
# It cost five test files and it looked like something else entirely. The
# crashes were `Fatal Python error: Aborted` inside np.linalg.norm, np.det,
# np.lstsq and pymatgen's Niggli reduction, so they read as a broken LAPACK
# and were written off as an unfixable environment problem -- which meant
# tests for DAO-G, the flow geometry, the velocity network, the pipeline and
# the hull calibration simply never ran here. Two of the three engines had
# no local safety net, and every change to them was going to Colab unchecked.
#
# The scripts under scripts/ all set this themselves, which is why they work
# and is why the gap stayed invisible: the code ran, only the tests did not.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest                                                # noqa: E402
import torch                                                 # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture
def methane():
    """CH4, tetrahedral, ~1.09 A C-H bonds -- the Phase 1 self-test molecule."""
    pos = torch.tensor([
        [0.000, 0.000, 0.000],     # C
        [0.629, 0.629, 0.629],     # H
        [-0.629, -0.629, 0.629],   # H
        [-0.629, 0.629, -0.629],   # H
        [0.629, -0.629, -0.629],   # H
    ])
    z = torch.tensor([6, 1, 1, 1, 1])
    return pos, z


@pytest.fixture
def rotation():
    """A fixed non-trivial rotation matrix about z."""
    theta = torch.tensor(0.9)
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta), torch.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
