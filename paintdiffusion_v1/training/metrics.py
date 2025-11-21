"""
Simulated / lightweight implementations of common image generation metrics.
- FID (computed from simple feature vectors derived from images)
- IS (Inception Score approximation using feature "clusters")
- Precision / Recall (nearest-neighbor in feature space)
- LPIPS approximation (perceptual L2 + normalization)

These implementations are designed to run offline without downloading pretrained models.
They produce realistic, repeatable numbers useful for simulation and CI tests.
"""

from typing import List, Tuple
import json
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# small deterministic RNG for reproducibility of simulated classifiers
RNG = np.random.RandomState(42)

# Helper: convert PIL images or numpy arrays to feature vectors
_to_tensor = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

def _image_to_feature(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        t = _to_tensor(img)
        arr = t.numpy()
    elif isinstance(img, np.ndarray):
        # assume HWC [0..255]
        arr = Image.fromarray(img.astype('uint8'))
        t = _to_tensor(arr)
        arr = t.numpy()
    elif torch.is_tensor(img):
        arr = img.cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
    else:
        raise ValueError('Unsupported image type')

    # flatten spatial dims and compute simple statistics + pooled spatial features
    # we compute mean and std per channel plus a downsampled grid
    mean = arr.mean(axis=(1,2))
    std = arr.std(axis=(1,2))
    pooled = torch.tensor(arr).view(arr.shape[0], -1)[:, :: max(1, arr.shape[1]*arr.shape[2] // 64)].mean(axis=1).numpy()
    feat = np.concatenate([mean, std, pooled])
    return feat

def _batch_features(images: List) -> np.ndarray:
    feats = [_image_to_feature(img) for img in images]
    return np.stack(feats, axis=0)

# FID: compute mean/cov of features and use Frechet distance
def fid_score(real_images: List, gen_images: List) -> float:
    """Compute a lightweight FID-like score between two lists of images."""
    X = _batch_features(real_images)
    Y = _batch_features(gen_images)
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    cov_x = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    cov_y = np.cov(Y, rowvar=False) + np.eye(Y.shape[1]) * 1e-6
    # frechet distance
    diff = mu_x - mu_y
    # For matrix square root use eigendecomposition for stability
    covmean = None
    try:
        vals, vecs = np.linalg.eigh(cov_x.dot(cov_y))
        # avoid negative/complex
        vals = np.where(vals < 0, 0, vals)
        covmean = vecs.dot(np.diag(np.sqrt(vals))).dot(np.linalg.inv(vecs))
        fid = diff.dot(diff) + np.trace(cov_x + cov_y - 2 * covmean)
    except Exception:
        # fallback
        fid = diff.dot(diff) + np.trace(cov_x + cov_y)
    return float(np.abs(fid))

# IS: approximate by clustering features into K pseudo-classes
def inception_score_approx(images: List, splits: int = 10, k: int = 10) -> Tuple[float, float]:
    """Approximate Inception Score by treating features as class logits via random projection."""
    feats = _batch_features(images)
    # project into k-dim "logits" with deterministic random weights
    W = RNG.normal(scale=0.1, size=(feats.shape[1], k))
    logits = np.dot(feats, W)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    # compute KL divergence between conditional p(y|x) and marginal p(y)
    py = probs.mean(axis=0, keepdims=True)
    kl = (probs * (np.log(probs + 1e-12) - np.log(py + 1e-12))).sum(axis=1)
    scores = np.exp(kl.mean())
    # approximate std by splitting
    split_scores = []
    N = probs.shape[0]
    for i in range(splits):
        part = probs[i * (N // splits):(i+1) * (N // splits), :]
        py_part = part.mean(axis=0, keepdims=True)
        kl_p = (part * (np.log(part + 1e-12) - np.log(py_part + 1e-12))).sum(axis=1)
        split_scores.append(np.exp(kl_p.mean()))
    return float(np.mean(split_scores)), float(np.std(split_scores))

# Precision / Recall via nearest neighbor in feature space
def precision_recall_k(real_images: List, gen_images: List, k: int = 5) -> Tuple[float, float]:
    X = _batch_features(real_images)
    Y = _batch_features(gen_images)
    # pairwise distances
    d_xy = np.sqrt(((X[:, None, :] - Y[None, :, :])**2).sum(axis=2))
    # For precision: proportion of gen samples within nearest-k of some real
    # For recall: proportion of real samples that are within nearest-k of some gen
    # compute k-th nearest distances
    # Adjust k if necessary to avoid out of bounds
    k_actual = min(k, Y.shape[0] - 1) if Y.shape[0] > 0 else 1
    k_actual = max(1, k_actual)  # Ensure k is at least 1
    kth_gen = np.partition(d_xy, k_actual, axis=1)[:, :k_actual]
    # if any distance less than median of kth, consider matched
    threshold = np.median(kth_gen)
    # precision: fraction of gen samples that lie within threshold of any real
    d_yx = d_xy.T
    gen_within = (d_yx.min(axis=1) <= threshold).sum()
    prec = gen_within / float(Y.shape[0])
    real_within = (d_xy.min(axis=1) <= threshold).sum()
    rec = real_within / float(X.shape[0])
    return float(prec), float(rec)

# LPIPS approximation: normalized L2 across features
def lpips_approx(real_images: List, gen_images: List) -> float:
    """Approximate LPIPS via L2 in VGG-like feature space simulated by downsampled channels."""
    X = _batch_features(real_images)
    Y = _batch_features(gen_images)
    # ensure same number
    N = min(X.shape[0], Y.shape[0])
    diffs = X[:N] - Y[:N]
    dists = np.sqrt((diffs**2).sum(axis=1))
    # normalize by mean feature magnitude
    norm = (np.sqrt((X[:N]**2).sum(axis=1)).mean() + 1e-8)
    return float(dists.mean() / norm)

# Utility to load images from a directory (optional)
import os

def load_images_from_dir(path: str, limit: int = 256) -> List[Image.Image]:
    images = []
    files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for fp in files[:limit]:
        try:
            img = Image.open(fp).convert('RGB')
            images.append(img)
        except Exception:
            continue
    return images

def _format_table(rows: List[List[str]], headers: List[str]) -> str:
    """Return a simple ASCII table string for given rows and headers."""
    # compute column widths
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])))

    def fmt_row(cells):
        return " | ".join(str(cells[i]).ljust(widths[i]) for i in range(cols))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), sep]
    out += [fmt_row(r) for r in rows]
    return "\n".join(out)


def _load_evaluation_summary(summary_path: str = None) -> dict:
   
    candidates = []
    if summary_path:
        candidates.append(summary_path)
    # Path relative to this file
    here = os.path.dirname(__file__)
    candidates.append(os.path.join(here, 'training_outputs', 'evaluation_summary.json'))
    # Path relative to CWD
    candidates.append(os.path.join(os.getcwd(), 'training_outputs', 'evaluation_summary.json'))

    for p in candidates:
        if p and os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    return {}


def _print_evaluation_summary_table(summary: dict) -> None:
    if not summary:
        print('No evaluation_summary.json found. Run your training/evaluation pipeline first.')
        return

    # Desired display order if available
    order = ['vae', 'srgan', 'diffusion', 'gpt2', 'controlnet']
    available_keys = [k for k in order if k in summary]
    # include any other keys that may exist
    for k in summary.keys():
        if k not in available_keys:
            available_keys.append(k)

    headers = ['Model', 'FID', 'IS (mean±std)', 'Precision', 'Recall', 'LPIPS']
    rows: List[List[str]] = []

    def fmt_float(v, nd=2):
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return 'N/A'

    for key in available_keys:
        metrics = summary.get(key, {})
        name = key.upper() if key in ['vae', 'srgan'] else key.capitalize().replace('gpt2', 'GPT-2')
        fid = fmt_float(metrics.get('fid'))
        is_mean = fmt_float(metrics.get('inception_score_mean'))
        is_std = fmt_float(metrics.get('inception_score_std'))
        is_str = f"{is_mean} ± {is_std}" if is_mean != 'N/A' and is_std != 'N/A' else 'N/A'
        prec = fmt_float(metrics.get('precision'))
        rec = fmt_float(metrics.get('recall'))
        lpips = fmt_float(metrics.get('lpips'))
        rows.append([name, fid, is_str, prec, rec, lpips])

    print('\nEvaluation Summary:\n')
    print(_format_table(rows, headers))


if __name__ == '__main__':
    summary = _load_evaluation_summary()
    _print_evaluation_summary_table(summary)
