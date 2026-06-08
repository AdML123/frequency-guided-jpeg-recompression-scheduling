"""DCT-frequency diagnostics for JPEG schedule selection."""

from __future__ import annotations

import numpy as np
import torch
from scipy.fft import dctn


_ELIMINATION_THRESHOLDS = {
    "cifar10": 2.73,
    "imagenet": 3.01,
}


def block_dct_coefficients(images, block_size=8):
    """Return blockwise 2-D DCT coefficients with JPEG-style 8x8 blocks."""
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size!r}")

    array = _to_numpy(images)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"images must have shape NxCxHxW or CxHxW, got {array.shape}")

    n_images, channels, height, width = array.shape
    cropped_height = height - (height % block_size)
    cropped_width = width - (width % block_size)
    if cropped_height == 0 or cropped_width == 0:
        raise ValueError(
            f"images must contain at least one {block_size}x{block_size} block, "
            f"got height={height}, width={width}"
        )

    cropped = array[:, :, :cropped_height, :cropped_width]
    blocks = cropped.reshape(
        n_images,
        channels,
        cropped_height // block_size,
        block_size,
        cropped_width // block_size,
        block_size,
    ).transpose(0, 1, 2, 4, 3, 5)
    return dctn(blocks, axes=(-2, -1), norm="ortho")


def frequency_energy_centroid(coefficients, block_size=8, eps=1e-12):
    """Return the radial DCT-frequency energy centroid for coefficients."""
    weighted_sum, energy_sum = frequency_energy_moments(
        coefficients,
        block_size=block_size,
    )
    return centroid_from_energy_moments(weighted_sum, energy_sum, eps=eps)


def frequency_energy_moments(coefficients, block_size=8):
    """Return radial weighted-energy and total-energy moments for coefficients."""
    coeff_array = _to_numpy(coefficients)
    if coeff_array.shape[-2:] != (int(block_size), int(block_size)):
        raise ValueError(
            "coefficients must end with block dimensions "
            f"{block_size}x{block_size}, got {coeff_array.shape}"
        )

    energy = np.square(coeff_array.astype(np.float64, copy=False))
    radius = frequency_radius_grid(block_size)
    return float((energy * radius).sum()), float(energy.sum())


def centroid_from_energy_moments(weighted_sum, energy_sum, eps=1e-12):
    """Return a centroid from weighted and total DCT-energy moments."""
    denominator = float(energy_sum)
    if denominator <= float(eps):
        return 0.0
    return float(weighted_sum) / denominator


def frequency_radius_grid(block_size=8):
    """Return a radial frequency grid for a DCT block."""
    positions = np.arange(int(block_size), dtype=np.float64)
    u, v = np.meshgrid(positions, positions, indexing="ij")
    return np.sqrt(np.square(u) + np.square(v))


def perturbation_frequency_centroid(clean_images, adversarial_images, block_size=8):
    """Measure omega_delta from the DCT energy of adversarial residuals."""
    clean = _to_numpy(clean_images)
    adversarial = _to_numpy(adversarial_images)
    if clean.shape != adversarial.shape:
        raise ValueError(
            f"clean and adversarial images must share shape, got {clean.shape} and {adversarial.shape}"
        )
    return frequency_energy_centroid(
        block_dct_coefficients(adversarial - clean, block_size=block_size),
        block_size=block_size,
    )


def elimination_threshold_for_dataset(dataset):
    """Return the dataset-specific omega_delta elimination threshold tau."""
    key = str(dataset).strip().lower().replace("-", "").replace("_", "")
    try:
        return _ELIMINATION_THRESHOLDS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_ELIMINATION_THRESHOLDS))
        raise ValueError(
            f"Unsupported dataset {dataset!r}; expected one of: {supported}"
        ) from exc


def omega_delta_relation(omega_delta, tau):
    """Return the paper-friendly omega_delta/tau relation."""
    return ">" if float(omega_delta) > float(tau) else "<="


def predict_schedule_from_omega_delta(omega_delta, tau, jpeg_aware_attack=False):
    """Return the threshold-rule JPEG schedule prediction."""
    high = float(omega_delta) > float(tau)
    if bool(jpeg_aware_attack):
        return "fixed" if high else "geometric"
    return "front_loaded" if high else "fixed"


def prediction_rule_label(jpeg_aware_attack=False, tau=None):
    """Return a compact human-readable rule label."""
    threshold = "tau" if tau is None else f"tau={float(tau):.2f}"
    if bool(jpeg_aware_attack):
        return f"JPEG-aware: fixed if omega_delta > {threshold} else geometric"
    return f"nonadaptive: front_loaded if omega_delta > {threshold} else fixed"


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
