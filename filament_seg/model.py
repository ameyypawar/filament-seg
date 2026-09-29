"""Segmentation network, loss, device selection and tiled inference.

A U-Net over a ResNet encoder is a deliberately boring choice: at native
2048x2048 resolution there is no budget for architecture search, and a
well-understood encoder/decoder pair with ImageNet-pretrained weights gives
the fastest path to a working model. The two-channel input (flattened
intensity + disk radius) needs its own first conv, so the pretrained weights
only warm-start the encoder trunk, not that very first layer.

``tiled_predict`` lives here rather than duplicated in ``scripts/train.py``
(per-epoch validation) and ``scripts/predict.py`` (submission generation) so
the two can never quietly drift apart on how tiles are blended.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import numpy as np
import torch
import torch.nn as nn


def build_model(
    encoder: str = "resnet34",
    in_channels: int = 2,
    weights: str | None = "imagenet",
) -> nn.Module:
    """U-Net over ``encoder``, one logit output channel (no activation).

    The loss (``DiceBCELoss``) applies ``BCEWithLogitsLoss`` internally, so
    the network stays activation-free -- this is also what keeps it
    numerically stable under autocast.
    """
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=1,
        activation=None,
    )


class DiceBCELoss(nn.Module):
    """Soft Dice + BCE-with-logits.

    Filament pixels are roughly 0.2% of the disk, so plain BCE is dominated
    by the easy background majority long before its gradient starts pushing
    on filament boundaries. Dice is defined on the foreground overlap and is
    immune to how much background surrounds it; BCE is kept alongside it
    because Dice alone gives a noisy, sometimes unstable signal on crops with
    very few (or zero) positive pixels.
    """

    def __init__(
        self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1.0
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)

        probs = torch.sigmoid(logits).flatten(1)
        target_flat = target.flatten(1)
        intersection = (probs * target_flat).sum(dim=1)
        union = probs.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def select_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _hann_window(size: int) -> np.ndarray:
    window = np.hanning(size)
    if window.sum() == 0:  # size == 1: np.hanning gives an all-zero window
        window = np.ones(size)
    return np.outer(window, window).astype(np.float32)


@torch.no_grad()
def tiled_predict(
    model: nn.Module,
    x: np.ndarray,
    tile: int = 512,
    overlap: int = 128,
    device: torch.device | None = None,
    batch_size: int = 4,
) -> np.ndarray:
    """Run ``model`` over overlapping tiles of a full-resolution image.

    Tiles are blended with a 2D Hann window rather than averaged uniformly: a
    tile's prediction is least reliable right at its own edge (no context
    beyond the tile boundary), so down-weighting the edges is what keeps the
    seams between tiles from fragmenting a filament that crosses one -- and
    fragmentation is punished roughly three times over under Panoptic
    Quality (see ``filament_seg.postprocess``).

    ``x`` is ``(C, H, W)`` float32; returns full-resolution logits ``(H, W)``.
    """
    model.eval()
    device = device or next(model.parameters()).device
    _, height, width = x.shape
    stride = max(tile - overlap, 1)

    def positions(extent: int) -> list[int]:
        starts = list(range(0, max(extent - tile, 0) + 1, stride))
        if not starts or starts[-1] != extent - tile:
            starts.append(max(extent - tile, 0))
        return starts

    ys, xs = positions(height), positions(width)
    window = _hann_window(tile)
    logits_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    grid = [(y, xc) for y in ys for xc in xs]
    for start in range(0, len(grid), batch_size):
        batch = grid[start : start + batch_size]
        tiles = np.stack([x[:, y : y + tile, xc : xc + tile] for y, xc in batch], axis=0)
        tensor = torch.from_numpy(tiles).to(device)
        out = model(tensor).squeeze(1).float().cpu().numpy()
        for (y, xc), tile_logits in zip(batch, out):
            logits_sum[y : y + tile, xc : xc + tile] += tile_logits * window
            weight_sum[y : y + tile, xc : xc + tile] += window

    weight_sum[weight_sum == 0] = 1.0
    return logits_sum / weight_sum


#: Test-time augmentation: which orientations to average over, each written as
#: (flip left-right, flip up-down, transpose). Training augmented with flips and
#: 90-degree rotations, so the model has seen all eight orientations of a square
#: image ("dihedral"); "flips" is the cheaper half of them.
TTA_MODES: dict[str, list[tuple[bool, bool, bool]]] = {
    "none": [(False, False, False)],
    "flips": [(False, False, False), (True, False, False),
              (False, True, False), (True, True, False)],
    "dihedral": [(flip_lr, flip_ud, transpose)
                 for transpose in (False, True)
                 for flip_lr in (False, True)
                 for flip_ud in (False, True)],
}


def _orient(a: np.ndarray, flip_lr: bool, flip_ud: bool, transpose: bool) -> np.ndarray:
    """Apply one orientation to the last two (spatial) axes."""
    if transpose:
        a = np.swapaxes(a, -1, -2)
    if flip_lr:
        a = a[..., :, ::-1]
    if flip_ud:
        a = a[..., ::-1, :]
    return np.ascontiguousarray(a)


def _unorient(a: np.ndarray, flip_lr: bool, flip_ud: bool, transpose: bool) -> np.ndarray:
    """Undo :func:`_orient` on a 2-D map -- the same steps, in reverse order."""
    if flip_ud:
        a = a[::-1, :]
    if flip_lr:
        a = a[:, ::-1]
    if transpose:
        a = a.T
    return a


@torch.no_grad()
def tiled_predict_tta(
    model: nn.Module, x: np.ndarray, tta: str = "none", **tiled_kwargs
) -> np.ndarray:
    """``tiled_predict`` averaged over mirrored and rotated copies of the image.

    Each copy is predicted, turned back to the original orientation, and the
    logits are averaged. The prediction for a thin filament then no longer
    depends on which way the tiles happened to cut across it. Both input
    channels -- intensity and the radius map -- are transformed together, so the
    model always sees a geometrically consistent image.

    Averaging makes logits less extreme than a single pass, so a threshold tuned
    without TTA is not automatically right with it.
    """
    orientations = TTA_MODES[tta]
    total: np.ndarray | None = None
    for orientation in orientations:
        logits = tiled_predict(model, _orient(x, *orientation), **tiled_kwargs)
        logits = _unorient(logits, *orientation)
        total = logits.copy() if total is None else total + logits
    return total / len(orientations)
