"""Test-time augmentation must map every prediction back to where it came from.

An inverse transform applied in the wrong order still produces a plausible
probability map -- just a blurred average of misaligned copies -- so the check
here is exact: with a model that simply echoes its input, averaging over any
set of orientations must return the input unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from filament_seg.model import TTA_MODES, _orient, _unorient, tiled_predict_tta


class EchoModel(nn.Module):
    """Returns the first input channel as its logits."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    # Deliberately asymmetric, so a wrong flip or transpose cannot cancel out.
    x = rng.normal(size=(2, 300, 300)).astype(np.float32)
    x[0, :40, :200] += 5.0
    return x


@pytest.mark.parametrize("mode", list(TTA_MODES))
def test_tta_returns_echo_model_input_exactly(image, mode):
    out = tiled_predict_tta(EchoModel(), image, tta=mode, tile=128, overlap=32,
                            device=torch.device("cpu"))
    assert out.shape == image.shape[1:]
    # The Hann window is zero on the outermost pixel ring, which tiled_predict
    # leaves at 0 regardless of the model; everything inside must match exactly.
    np.testing.assert_allclose(out[1:-1, 1:-1], image[0, 1:-1, 1:-1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("orientation", TTA_MODES["dihedral"])
def test_each_orientation_round_trips(image, orientation):
    oriented = _orient(image, *orientation)
    assert oriented.flags["C_CONTIGUOUS"], "torch.from_numpy needs contiguous input"
    np.testing.assert_array_equal(_unorient(oriented[0], *orientation), image[0])


def test_dihedral_covers_eight_distinct_orientations(image):
    seen = {_orient(image[0], *o).tobytes() for o in TTA_MODES["dihedral"]}
    assert len(seen) == 8
