"""
Evaluate a pair of image sets (real vs generated) using the metrics in metrics.py
Produces a JSON summary with FID, IS, Precision/Recall and LPIPS.
"""

import os
import json
from typing import List
from PIL import Image
from metrics import fid_score, inception_score_approx, precision_recall_k, lpips_approx, load_images_from_dir


def evaluate(real_images: List[Image.Image], gen_images: List[Image.Image]) -> dict:
    results = {}
    # Ensure non-empty
    if len(real_images) == 0 or len(gen_images) == 0:
        raise ValueError('Both real_images and gen_images must be non-empty lists of PIL images')

    results['fid'] = fid_score(real_images, gen_images)
    is_mean, is_std = inception_score_approx(gen_images)
    results['inception_score_mean'] = is_mean
    results['inception_score_std'] = is_std
    prec, rec = precision_recall_k(real_images, gen_images)
    results['precision'] = prec
    results['recall'] = rec
    results['lpips'] = lpips_approx(real_images, gen_images)
    return results


def evaluate_dirs(real_dir: str, gen_dir: str, limit: int = 256) -> dict:
    real_images = load_images_from_dir(real_dir, limit=limit)
    gen_images = load_images_from_dir(gen_dir, limit=limit)
    return evaluate(real_images, gen_images)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', required=True, help='Directory of real images')
    parser.add_argument('--gen_dir', required=True, help='Directory of generated images')
    parser.add_argument('--out', default='metrics_report.json')
    args = parser.parse_args()

    res = evaluate_dirs(args.real_dir, args.gen_dir)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print('Metrics saved to', args.out)
