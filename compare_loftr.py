#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append('core')

from raft_stereo import RAFTStereo
from utils.frame_utils import readDispKITTI, readDispMiddlebury, readPFM
from utils.utils import InputPadder


LOGGER = logging.getLogger("raftstereo.compare_loftr")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(imfile: str) -> Tuple[np.ndarray, torch.Tensor]:
    image = np.array(Image.open(imfile).convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    return image, tensor


def load_checkpoint(model: torch.nn.Module, restore_ckpt: str, use_dense_frontend: bool):
    checkpoint = torch.load(restore_ckpt, map_location="cpu")
    strict = not use_dense_frontend
    try:
        result = model.load_state_dict(checkpoint, strict=strict)
    except RuntimeError:
        if isinstance(checkpoint, dict) and checkpoint and all(key.startswith('module.') for key in checkpoint.keys()):
            stripped = {key[len('module.'):]: value for key, value in checkpoint.items()}
            result = model.load_state_dict(stripped, strict=strict)
        else:
            raise
    if strict:
        return

    missing = [key for key in result.missing_keys if 'dense_matcher' not in key]
    unexpected = [key for key in result.unexpected_keys if 'dense_matcher' not in key]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing}. Unexpected: {unexpected}")

    if result.missing_keys:
        LOGGER.warning("Dense frontend keys are not stored in the checkpoint; using the module's current weights.")


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="alt",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2,
                        help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch",
                        choices=['group', 'batch', 'instance', 'none'],
                        help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during forward pass')



def add_dense_args(parser: argparse.ArgumentParser):
    parser.add_argument('--dense_frontend_pretrained', type=str, default='outdoor',
                        help='pretrained preset for the selected frontend matcher')
    parser.add_argument('--dense_frontend_confidence_thresh', type=float, default=0.2,
                        help='minimum confidence kept from LoFTR/RoMa matches')
    parser.add_argument('--dense_frontend_max_vertical_offset', type=float, default=2.0,
                        help='maximum allowed vertical mismatch in pixels for LoFTR/RoMa matches')
    parser.add_argument('--dense_frontend_blur_kernel', type=int, default=7,
                        help='odd kernel size used to spread sparse matches onto the RAFT grid')
    parser.add_argument('--dense_frontend_blur_std', type=float, default=2.0,
                        help='gaussian std used to spread sparse matches onto the RAFT grid')
    parser.add_argument('--dense_frontend_trainable', action='store_true',
                        help='allow gradients through the selected external frontend when supported')


def build_args(base_args: argparse.Namespace, use_dense_frontend: bool):
    args = deepcopy(base_args)
    args.dataset_file = 'rsvg'
    args.binary = True
    args.with_box_refine = True
    args.num_frames = 1
    args.backbone = 'resnet50'
    args.num_feature_levels = 4
    args.use_dense_frontend = use_dense_frontend
    args.dense_frontend_type = 'loftr'
    args.device = DEVICE.type
    return args


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    model = RAFTStereo(args)
    if DEVICE.type == 'cuda':
        device_id = DEVICE.index if DEVICE.index is not None else 0
        model = torch.nn.DataParallel(model, device_ids=[device_id])
    model.to(DEVICE)
    model.eval()
    return model


def run_inference(model: torch.nn.Module, left_tensor: torch.Tensor, right_tensor: torch.Tensor, iters: int):
    padder = InputPadder(left_tensor.shape, divis_by=32)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    started_at = time.perf_counter()
    with torch.inference_mode():
        _, flow_up = model(left_tensor, right_tensor, iters=iters, test_mode=True)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - started_at) * 1000.0

    return padder.unpad(flow_up).squeeze(0).squeeze(0).detach().cpu().numpy(), runtime_ms


def collect_paths(left_pattern: str, right_pattern: str, gt_pattern: Optional[str], max_pairs: int = 0):
    left_paths = sorted(glob.glob(left_pattern, recursive=True))
    right_paths = sorted(glob.glob(right_pattern, recursive=True))
    if len(left_paths) != len(right_paths):
        raise ValueError(f"Left/right image counts differ: {len(left_paths)} vs {len(right_paths)}")

    gt_paths = None
    if gt_pattern:
        gt_paths = sorted(glob.glob(gt_pattern, recursive=True))
        if len(gt_paths) != len(left_paths):
            raise ValueError(f"GT/image counts differ: {len(gt_paths)} vs {len(left_paths)}")

    pairs = []
    seen_names: Dict[str, int] = {}
    for idx, (left_path, right_path) in enumerate(zip(left_paths, right_paths)):
        if max_pairs and idx >= max_pairs:
            break
        if gt_paths is None:
            gt_path = None
        else:
            gt_path = gt_paths[idx]

        candidate = Path(left_path).parent.name or Path(left_path).stem or f"pair_{idx:04d}"
        seen_names[candidate] = seen_names.get(candidate, 0) + 1
        if seen_names[candidate] > 1:
            candidate = f"{candidate}_{seen_names[candidate]}"

        pairs.append(
            {
                'name': candidate,
                'left_path': left_path,
                'right_path': right_path,
                'gt_path': gt_path,
            }
        )

    return pairs


def load_gt_disparity(gt_path: str):
    suffix = Path(gt_path).suffix.lower()
    name = Path(gt_path).name
    if suffix == '.pfm':
        if name in {'disp0GT.pfm', 'disp0.pfm'}:
            disp, valid = readDispMiddlebury(gt_path)
            return disp.astype(np.float32), valid.astype(bool)
        disp = readPFM(gt_path).astype(np.float32)
        valid = np.isfinite(disp) & (disp > 0)
        return disp, valid
    if suffix == '.png':
        disp, valid = readDispKITTI(gt_path)
        return disp.astype(np.float32), valid.astype(bool)
    if suffix == '.npy':
        disp = np.load(gt_path).astype(np.float32)
        valid = np.isfinite(disp)
        return disp, valid
    raise ValueError(f"Unsupported GT disparity format: {gt_path}")


def compute_error_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    mask = np.isfinite(pred) & np.isfinite(gt) & valid
    if not np.any(mask):
        return {
            'mae_px': float('nan'),
            'bad1_pct': float('nan'),
            'bad3_pct': float('nan'),
            'valid_pixels': 0,
        }

    error = np.abs(pred[mask] - gt[mask])
    return {
        'mae_px': float(error.mean()),
        'bad1_pct': float((error > 1.0).mean() * 100.0),
        'bad3_pct': float((error > 3.0).mean() * 100.0),
        'valid_pixels': int(mask.sum()),
    }


def compute_diff_stats(baseline: np.ndarray, loftr: np.ndarray):
    mask = np.isfinite(baseline) & np.isfinite(loftr)
    if not np.any(mask):
        return {
            'mean_abs_diff': float('nan'),
            'median_abs_diff': float('nan'),
            'max_abs_diff': float('nan'),
        }

    diff = np.abs(baseline[mask] - loftr[mask])
    return {
        'mean_abs_diff': float(diff.mean()),
        'median_abs_diff': float(np.median(diff)),
        'max_abs_diff': float(diff.max()),
    }


def shared_disparity_range(arrays: Sequence[np.ndarray]):
    values = []
    for array in arrays:
        mask = np.isfinite(array)
        if np.any(mask):
            values.append(array[mask].reshape(-1))
    if not values:
        return 0.0, 1.0

    merged = np.concatenate(values)
    if merged.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(merged, 2))
    vmax = float(np.percentile(merged, 98))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def diff_range(array: np.ndarray):
    mask = np.isfinite(array)
    if not np.any(mask):
        return 0.0, 1.0
    values = array[mask]
    vmax = float(np.percentile(values, 98))
    if vmax <= 0:
        vmax = float(values.max()) if values.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    return 0.0, vmax


def save_disparity_png(array: np.ndarray, path: Path, vmin: float, vmax: float, cmap: str):
    plt.imsave(path, array, cmap=cmap, vmin=vmin, vmax=vmax)


def render_comparison_figure(
    pair_name: str,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    baseline_disp: np.ndarray,
    loftr_disp: np.ndarray,
    baseline_runtime_ms: float,
    loftr_runtime_ms: float,
    output_path: Path,
    gt_disp: Optional[np.ndarray] = None,
    gt_valid: Optional[np.ndarray] = None,
    baseline_metrics: Optional[Dict[str, float]] = None,
    loftr_metrics: Optional[Dict[str, float]] = None,
):
    disp_arrays = [baseline_disp, loftr_disp]
    if gt_disp is not None:
        disp_arrays.append(gt_disp)
    vmin, vmax = shared_disparity_range(disp_arrays)

    diff_map = np.abs(baseline_disp - loftr_disp)
    diff_valid = diff_map[np.isfinite(diff_map)]
    diff_mean = float(diff_valid.mean()) if diff_valid.size else float('nan')
    diff_vmin, diff_vmax = diff_range(diff_map)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5), constrained_layout=True)
    fig.suptitle(pair_name, fontsize=14)

    axes[0].imshow(left_rgb)
    axes[0].set_title("Left image")
    axes[1].imshow(right_rgb)
    axes[1].set_title("Right image")

    baseline_title = f"Baseline\n{baseline_runtime_ms:.1f} ms"
    loftr_title = f"LoFTR\n{loftr_runtime_ms:.1f} ms"
    if baseline_metrics is not None and loftr_metrics is not None:
        baseline_title += f"\nMAE {baseline_metrics['mae_px']:.2f} px | Bad1 {baseline_metrics['bad1_pct']:.1f}%"
        loftr_title += f"\nMAE {loftr_metrics['mae_px']:.2f} px | Bad1 {loftr_metrics['bad1_pct']:.1f}%"
    axes[2].imshow(baseline_disp, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[2].set_title(baseline_title)
    axes[3].imshow(loftr_disp, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[3].set_title(loftr_title)
    axes[4].imshow(diff_map, cmap='magma', vmin=diff_vmin, vmax=diff_vmax)
    axes[4].set_title(f"|Baseline - LoFTR|\nmean {diff_mean:.2f} px")

    for axis in axes:
        axis.axis('off')

    fig.savefig(output_path, dpi=180, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def infer_pass(
    model: torch.nn.Module,
    pairs: Sequence[Dict[str, str]],
    iters: int,
    pass_name: str,
):
    results = []
    for pair in tqdm(pairs, desc=pass_name):
        left_rgb, left_tensor = load_image(pair['left_path'])
        right_rgb, right_tensor = load_image(pair['right_path'])
        disparity, runtime_ms = run_inference(model, left_tensor, right_tensor, iters)
        results.append(
            {
                'name': pair['name'],
                'left_rgb': left_rgb,
                'right_rgb': right_rgb,
                'disparity': disparity,
                'runtime_ms': runtime_ms,
                'left_path': pair['left_path'],
                'right_path': pair['right_path'],
                'gt_path': pair['gt_path'],
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare RAFT-Stereo baseline vs LoFTR frontend")
    parser.add_argument('--restore_ckpt', required=True, help='checkpoint path')
    parser.add_argument('-l', '--left_imgs', default='datasets/Middlebury/MiddEval3/testF/*/im0.png',
                        help='glob for left images')
    parser.add_argument('-r', '--right_imgs', default='datasets/Middlebury/MiddEval3/testF/*/im1.png',
                        help='glob for right images')
    parser.add_argument('--gt_disparities', default=None,
                        help='optional glob for GT disparity files (PFM / KITTI PNG / NPY)')
    parser.add_argument('--output_directory', default='compare_output', help='directory to save outputs')
    parser.add_argument('--max_pairs', type=int, default=0, help='limit number of pairs processed (0 = all)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='torch device')

    add_model_args(parser)
    add_dense_args(parser)
    args = parser.parse_args()

    global DEVICE
    DEVICE = torch.device(args.device)
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

    pairs = collect_paths(args.left_imgs, args.right_imgs, args.gt_disparities, args.max_pairs)
    if not pairs:
        raise ValueError('No stereo pairs matched the supplied patterns.')

    output_root = Path(args.output_directory)
    baseline_dir = output_root / 'baseline'
    loftr_dir = output_root / 'loftr'
    compare_dir = output_root / 'compare'
    baseline_dir.mkdir(parents=True, exist_ok=True)
    loftr_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    baseline_args = build_args(args, use_dense_frontend=False)
    loftr_args = build_args(args, use_dense_frontend=True)

    LOGGER.info('Running baseline inference on %d pair(s)', len(pairs))
    baseline_model = build_model(baseline_args)
    load_checkpoint(baseline_model, args.restore_ckpt, use_dense_frontend=False)
    baseline_results = infer_pass(baseline_model, pairs, args.valid_iters, 'baseline')

    del baseline_model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    LOGGER.info('Running LoFTR inference on %d pair(s)', len(pairs))
    loftr_model = build_model(loftr_args)
    load_checkpoint(loftr_model, args.restore_ckpt, use_dense_frontend=True)
    loftr_results = infer_pass(loftr_model, pairs, args.valid_iters, 'loftr')

    del loftr_model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    summary_rows = []
    for pair, baseline, loftr in zip(pairs, baseline_results, loftr_results):
        gt_disp = None
        gt_valid = None
        baseline_metrics = None
        loftr_metrics = None
        if pair['gt_path']:
            gt_disp, gt_valid = load_gt_disparity(pair['gt_path'])
            baseline_metrics = compute_error_metrics(baseline['disparity'], gt_disp, gt_valid)
            loftr_metrics = compute_error_metrics(loftr['disparity'], gt_disp, gt_valid)

        diff_stats = compute_diff_stats(baseline['disparity'], loftr['disparity'])
        render_comparison_figure(
            pair_name=pair['name'],
            left_rgb=baseline['left_rgb'],
            right_rgb=baseline['right_rgb'],
            baseline_disp=baseline['disparity'],
            loftr_disp=loftr['disparity'],
            baseline_runtime_ms=baseline['runtime_ms'],
            loftr_runtime_ms=loftr['runtime_ms'],
            output_path=compare_dir / f"{pair['name']}.png",
            gt_disp=gt_disp,
            gt_valid=gt_valid,
            baseline_metrics=baseline_metrics,
            loftr_metrics=loftr_metrics,
        )

        disp_arrays = [baseline['disparity'], loftr['disparity']]
        if gt_disp is not None:
            disp_arrays.append(gt_disp)
        vmin, vmax = shared_disparity_range(disp_arrays)
        save_disparity_png(baseline['disparity'], baseline_dir / f"{pair['name']}.png", vmin, vmax, 'turbo')
        save_disparity_png(loftr['disparity'], loftr_dir / f"{pair['name']}.png", vmin, vmax, 'turbo')
        save_disparity_png(np.abs(baseline['disparity'] - loftr['disparity']), compare_dir / f"{pair['name']}_absdiff.png", *diff_range(np.abs(baseline['disparity'] - loftr['disparity'])), 'magma')

        LOGGER.info(
            '[%s] baseline %.1f ms | LoFTR %.1f ms | mean |Δ| %.3f px%s',
            pair['name'],
            baseline['runtime_ms'],
            loftr['runtime_ms'],
            diff_stats['mean_abs_diff'],
            '' if baseline_metrics is None else f" | MAE {baseline_metrics['mae_px']:.2f}->{loftr_metrics['mae_px']:.2f} px",
        )

        row = {
            'name': pair['name'],
            'left_path': pair['left_path'],
            'right_path': pair['right_path'],
            'gt_path': pair['gt_path'] or '',
            'baseline_runtime_ms': round(baseline['runtime_ms'], 3),
            'loftr_runtime_ms': round(loftr['runtime_ms'], 3),
            'mean_abs_diff_px': round(diff_stats['mean_abs_diff'], 4),
            'median_abs_diff_px': round(diff_stats['median_abs_diff'], 4),
            'max_abs_diff_px': round(diff_stats['max_abs_diff'], 4),
        }
        if baseline_metrics is not None and loftr_metrics is not None:
            row.update(
                {
                    'baseline_mae_px': round(baseline_metrics['mae_px'], 4),
                    'baseline_bad1_pct': round(baseline_metrics['bad1_pct'], 4),
                    'baseline_bad3_pct': round(baseline_metrics['bad3_pct'], 4),
                    'loftr_mae_px': round(loftr_metrics['mae_px'], 4),
                    'loftr_bad1_pct': round(loftr_metrics['bad1_pct'], 4),
                    'loftr_bad3_pct': round(loftr_metrics['bad3_pct'], 4),
                    'delta_mae_px': round(baseline_metrics['mae_px'] - loftr_metrics['mae_px'], 4),
                    'delta_bad1_pct': round(baseline_metrics['bad1_pct'] - loftr_metrics['bad1_pct'], 4),
                    'delta_bad3_pct': round(baseline_metrics['bad3_pct'] - loftr_metrics['bad3_pct'], 4),
                }
            )
        summary_rows.append(row)

    summary_path = output_root / 'summary.csv'
    fieldnames = [
        'name', 'left_path', 'right_path', 'gt_path',
        'baseline_runtime_ms', 'loftr_runtime_ms',
        'mean_abs_diff_px', 'median_abs_diff_px', 'max_abs_diff_px',
        'baseline_mae_px', 'baseline_bad1_pct', 'baseline_bad3_pct',
        'loftr_mae_px', 'loftr_bad1_pct', 'loftr_bad3_pct',
        'delta_mae_px', 'delta_bad1_pct', 'delta_bad3_pct',
    ]
    with summary_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})

    manifest_path = output_root / 'manifest.json'
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump({'pairs': summary_rows}, handle, indent=2, ensure_ascii=False)

    print(f'Saved compare figures to: {compare_dir}')
    print(f'Saved baseline disparity maps to: {baseline_dir}')
    print(f'Saved LoFTR disparity maps to: {loftr_dir}')
    print(f'Summary CSV: {summary_path}')


if __name__ == '__main__':
    main()
