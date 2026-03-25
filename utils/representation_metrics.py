import math

import torch


def center_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(features.shape)}")
    return features - features.mean(dim=0, keepdim=True)


def covariance_matrix(features: torch.Tensor, *, center: bool = True, eps: float = 1e-12) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(features.shape)}")
    x = center_features(features) if center else features
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples to estimate covariance")
    return (x.T @ x) / max(n - 1, 1) + eps * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)


def spectral_profile_from_covariance(covariance: torch.Tensor, eps: float = 1e-12) -> dict:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"Expected a square covariance matrix, got shape {tuple(covariance.shape)}")

    eigvals = torch.linalg.eigvalsh(covariance).clamp_min(eps)
    eigvals = torch.flip(eigvals, dims=[0])
    total = eigvals.sum().clamp_min(eps)
    probs = (eigvals / total).clamp_min(eps)

    entropy = -(probs * probs.log()).sum()
    normalized_entropy = entropy / math.log(float(probs.numel())) if probs.numel() > 1 else torch.zeros((), device=covariance.device, dtype=covariance.dtype)
    effective_rank = entropy.exp()
    participation_ratio = (total * total) / eigvals.square().sum().clamp_min(eps)
    top1_share = eigvals[0] / total
    topk_share = probs[: min(10, probs.numel())].sum()
    condition_number = eigvals[0] / eigvals[-1].clamp_min(eps)

    return {
        "num_samples": None,
        "num_features": int(covariance.shape[0]),
        "spectral_entropy": float(entropy.item()),
        "normalized_spectral_entropy": float(normalized_entropy.item()),
        "effective_rank": float(effective_rank.item()),
        "participation_ratio": float(participation_ratio.item()),
        "top1_share": float(top1_share.item()),
        "top10_share": float(topk_share.item()),
        "condition_number": float(condition_number.item()),
        "eigenvalues": eigvals.detach().cpu(),
    }


def isotropy_metrics(features: torch.Tensor) -> dict:
    cov = covariance_matrix(features)
    metrics = spectral_profile_from_covariance(cov)
    metrics["num_samples"] = int(features.shape[0])
    return metrics


def stack_activation_batches(batches: list[torch.Tensor]) -> torch.Tensor:
    if not batches:
        raise ValueError("No activation batches were collected")
    flattened = []
    feature_dim = None
    for batch in batches:
        if batch.ndim != 3:
            raise ValueError(f"Expected activation tensor with shape (B, T, C), got {tuple(batch.shape)}")
        batch_2d = batch.reshape(-1, batch.shape[-1])
        feature_dim = batch_2d.shape[-1] if feature_dim is None else feature_dim
        if batch_2d.shape[-1] != feature_dim:
            raise ValueError("Activation batches have mismatched feature dimensions")
        flattened.append(batch_2d)
    return torch.cat(flattened, dim=0)
