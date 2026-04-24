#!/usr/bin/env python3
"""Generate a static prototype interpretability atlas for PNP-DINOv2.

This script is intentionally GPU-oriented. It scans CUB images, finds the top
activating real image regions for every class prototype, optionally labels those
regions with CLIP, and writes browser-friendly static assets.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT / 'proto_non_param'
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded, DINOBackboneExpanded  # noqa: E402
from modeling.pnp import PCA, PNP  # noqa: E402

LOGGER = logging.getLogger('interpretability_report')

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_CLIP_CACHE = None

PART_PROMPTS = [
    'bird head', 'bird eye', 'bird beak', 'bird throat', 'bird crown',
    'bird breast', 'bird belly', 'bird back', 'bird wing', 'bird tail',
    'bird leg', 'bird foot', 'feather pattern', 'wing bar', 'body silhouette',
    'color patch', 'neck', 'rump', 'primary feathers', 'spotted plumage',
]

def load_part_metadata(dataset_root: Path) -> dict[str, Any]:
    cub_root = dataset_root / 'CUB_200_2011'
    parts = {}
    for line in (cub_root / 'parts' / 'parts.txt').read_text(encoding='utf-8').splitlines():
        part_id, part_name = line.split(' ', 1)
        parts[int(part_id)] = part_name.strip().replace('_', ' ')

    images = pd.read_csv(cub_root / 'images.txt', sep=' ', names=['image_id', 'image_path'])
    bboxes = pd.read_csv(cub_root / 'bounding_boxes.txt', sep=' ', names=['image_id', 'x', 'y', 'w', 'h'])
    part_locs = pd.read_csv(
        cub_root / 'parts' / 'part_locs.txt',
        sep=' ',
        names=['image_id', 'part_id', 'x', 'y', 'visible'],
    )

    image_to_id = dict(zip(images.image_path, images.image_id))
    bbox_by_id = {int(row.image_id): row for row in bboxes.itertuples(index=False)}
    parts_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in part_locs.itertuples(index=False):
        if int(row.visible) != 1:
            continue
        bbox = bbox_by_id.get(int(row.image_id))
        if bbox is None or float(bbox.w) <= 0 or float(bbox.h) <= 0:
            continue
        x_crop = (float(row.x) - float(bbox.x)) / float(bbox.w) * 224.0
        y_crop = (float(row.y) - float(bbox.y)) / float(bbox.h) * 224.0
        parts_by_id[int(row.image_id)].append({
            'part_id': int(row.part_id),
            'part_name': parts.get(int(row.part_id), f'part {int(row.part_id)}'),
            'x': x_crop,
            'y': y_crop,
        })

    return {
        'part_names': parts,
        'image_to_id': image_to_id,
        'parts_by_id': parts_by_id,
    }


def activation_box_from_map(activation: torch.Tensor, image_size: tuple[int, int] = (224, 224), half_size: int = 36) -> dict[str, int]:
    heat = tensor_to_heatmap(activation, image_size)
    flat_idx = int(np.argmax(heat))
    y, x = divmod(flat_idx, heat.shape[1])
    return {
        'left': max(0, x - half_size),
        'top': max(0, y - half_size),
        'right': min(image_size[0], x + half_size),
        'bottom': min(image_size[1], y + half_size),
    }


def vote_part_label(items: list[tuple[float, str, torch.Tensor]], part_metadata: dict[str, Any], dataset_root: Path, half_size: int = 36) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]]]:
    image_to_id = part_metadata['image_to_id']
    parts_by_id = part_metadata['parts_by_id']
    weighted_votes: Counter[str] = Counter()
    count_votes: Counter[str] = Counter()
    evidence = []

    for score, image_path, activation in items:
        rel_path = Path(image_path).relative_to(dataset_root / 'cub200_cropped' / 'test_cropped').as_posix()
        image_id = image_to_id.get(rel_path)
        if image_id is None:
            continue
        box = activation_box_from_map(activation, half_size=half_size)
        matched = []
        for part in parts_by_id.get(int(image_id), []):
            inside = box['left'] <= part['x'] <= box['right'] and box['top'] <= part['y'] <= box['bottom']
            if not inside:
                continue
            part_name = part['part_name']
            weighted_votes[part_name] += float(score)
            count_votes[part_name] += 1
            matched.append(part_name)
        evidence.append({
            'image_id': int(image_id),
            'activation_score': float(score),
            'box': box,
            'matched_parts': matched,
        })

    if not weighted_votes:
        return 'part unclear', 0.0, evidence, []

    label, vote_score = weighted_votes.most_common(1)[0]
    total = sum(weighted_votes.values()) or 1.0
    confidence = float(vote_score / total)
    votes = [
        {
            'label': part_name,
            'weighted_score': float(score),
            'share': float(score / total),
            'count': int(count_votes[part_name]),
        }
        for part_name, score in weighted_votes.most_common()
    ]
    return label, confidence, evidence, votes


def activation_center_from_map(activation: torch.Tensor, image_size: tuple[int, int] = (224, 224)) -> tuple[float, float]:
    heat = tensor_to_heatmap(activation, image_size)
    flat_idx = int(np.argmax(heat))
    y, x = divmod(flat_idx, heat.shape[1])
    return float(x), float(y)


def select_vote_items(
    items: list[tuple[float, str, torch.Tensor]],
    percentile: float,
    min_items: int,
) -> list[tuple[float, str, torch.Tensor]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: item[0], reverse=True)
    scores = np.asarray([item[0] for item in ordered], dtype=np.float32)
    threshold = float(np.percentile(scores, percentile))
    selected = [item for item in ordered if float(item[0]) >= threshold]
    if min_items > 0 and len(selected) < min(min_items, len(ordered)):
        selected = ordered[:min(min_items, len(ordered))]
    return selected


def vote_part_label_gaussian(
    items: list[tuple[float, str, torch.Tensor]],
    part_metadata: dict[str, Any],
    dataset_root: Path,
    image_root: Path,
    half_size: int = 36,
    vote_percentile: float = 75.0,
    min_vote_items: int = 5,
) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    image_to_id = part_metadata['image_to_id']
    parts_by_id = part_metadata['parts_by_id']
    selected = select_vote_items(items, vote_percentile, min_vote_items)
    weighted_votes: Counter[str] = Counter()
    count_votes: Counter[str] = Counter()
    evidence: list[dict[str, Any]] = []
    sigma = (2.0 * half_size * math.sqrt(2.0)) / 2.0

    for score, image_path, activation in selected:
        try:
            rel_path = Path(image_path).relative_to(image_root).as_posix()
        except ValueError:
            rel_path = Path(image_path).name
        image_id = image_to_id.get(rel_path)
        if image_id is None:
            continue

        cx, cy = activation_center_from_map(activation)
        box = activation_box_from_map(activation, half_size=half_size)
        parts = parts_by_id.get(int(image_id), [])
        part_rows = []
        gaussian_values = []
        for part in parts:
            dx = float(part['x']) - cx
            dy = float(part['y']) - cy
            distance = math.sqrt(dx * dx + dy * dy)
            gaussian = math.exp(-(distance * distance) / (2.0 * sigma * sigma))
            gaussian_values.append(gaussian)
            part_rows.append({
                'label': part['part_name'],
                'part_id': int(part['part_id']),
                'x': float(part['x']),
                'y': float(part['y']),
                'distance': float(distance),
                'gaussian': float(gaussian),
            })

        denom = sum(gaussian_values)
        if denom <= 0:
            continue

        for row in part_rows:
            normalized = float(row['gaussian'] / denom)
            vote = float(score) * normalized
            row['normalized_gaussian'] = normalized
            row['vote'] = vote
            weighted_votes[row['label']] += vote
            count_votes[row['label']] += 1

        evidence.append({
            'image_id': int(image_id),
            'activation_score': float(score),
            'box': box,
            'activation_center': {'x': cx, 'y': cy},
            'sigma': float(sigma),
            'part_votes': sorted(part_rows, key=lambda row: row['vote'], reverse=True),
            'matched_parts': [row['label'] for row in sorted(part_rows, key=lambda row: row['vote'], reverse=True)[:3]],
        })

    if not weighted_votes:
        return 'part unclear', 0.0, evidence, [], {
            'method': 'owner-class percentile gaussian proximity voting',
            'percentile': float(vote_percentile),
            'retained_count': len(selected),
            'scanned_owner_class_count': len(items),
            'sigma_rule': 'activation box diagonal / 2',
            'sigma': float(sigma),
        }

    label, vote_score = weighted_votes.most_common(1)[0]
    total = sum(weighted_votes.values()) or 1.0
    confidence = float(vote_score / total)
    votes = [
        {
            'label': part_name,
            'weighted_score': float(score),
            'share': float(score / total),
            'count': int(count_votes[part_name]),
        }
        for part_name, score in weighted_votes.most_common()
    ]
    process = {
        'method': 'owner-class percentile gaussian proximity voting',
        'percentile': float(vote_percentile),
        'activation_cutoff': f'top {100.0 - float(vote_percentile):.1f}% owner-class activations',
        'retained_count': len(selected),
        'scanned_owner_class_count': len(items),
        'sigma_rule': 'activation box diagonal / 2',
        'sigma': float(sigma),
        'votes': votes,
        'evidence': evidence[:10],
    }
    return label, confidence, evidence, votes, process

@dataclass
class Candidate:
    score: float
    image_path: str
    target: int
    overlay_path: str
    original_path: str
    crop_path: str
    activation_path: str


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def safe_name(value: str) -> str:
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value.strip())
    return value.strip('_') or 'unknown'


def load_classes(dataset_root: Path) -> list[str]:
    class_file = dataset_root / 'CUB_200_2011' / 'classes.txt'
    if class_file.exists():
        names = []
        for line in class_file.read_text(encoding='utf-8').splitlines():
            _, name = line.split(' ', 1)
            names.append(name.replace('_', ' '))
        return names
    folder_root = dataset_root / 'cub200_cropped' / 'test_cropped'
    return [p.name.replace('_', ' ') for p in sorted(folder_root.iterdir()) if p.is_dir()]


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> tuple[PNP, argparse.Namespace, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    hparams = argparse.Namespace(**ckpt['hparams'])
    state_dict = ckpt['state_dict']

    if 'dinov2' in hparams.backbone:
        if hparams.num_splits and hparams.num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=hparams.backbone,
                n_splits=hparams.num_splits,
                mode='block_expansion',
                freeze_norm_layer=True,
            )
        else:
            backbone = DINOv2Backbone(name=hparams.backbone)
        dim = backbone.dim
    elif 'dino' in hparams.backbone:
        backbone = DINOBackboneExpanded(
            name=hparams.backbone,
            n_splits=hparams.num_splits,
            mode='block_expansion',
            freeze_norm_layer=True,
        )
        dim = backbone.dim
    else:
        raise NotImplementedError(f'Unsupported backbone: {hparams.backbone}')

    n_classes = 200
    fg_extractor = PCA(bg_class=n_classes, compare_fn='le', threshold=0.5)
    model = PNP(
        backbone=backbone,
        dim=dim,
        fg_extractor=fg_extractor,
        n_prototypes=hparams.num_prototypes,
        n_classes=n_classes,
        gamma=hparams.gamma,
        temperature=hparams.temperature,
        sa_init=hparams.sa_initial_value,
        use_sinkhorn=True,
        norm_prototypes=False,
    )
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, hparams, ckpt


def parse_metrics(log_path: Path) -> dict[str, float | None]:
    metrics = {
        'accuracy': None,
        'consistency': None,
        'stability': None,
        'distinctiveness': None,
        'comprehensiveness': None,
    }
    if not log_path.exists():
        return metrics
    for line in log_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if 'Accuracy:' in line:
            metrics['accuracy'] = float(line.rsplit('Accuracy:', 1)[1].strip())
        elif 'Network consistency score:' in line:
            metrics['consistency'] = float(line.rsplit(':', 1)[1].strip())
        elif 'Network stability score:' in line:
            metrics['stability'] = float(line.rsplit(':', 1)[1].strip())
        elif 'Distinctiveness Score' in line:
            metrics['distinctiveness'] = float(line.rsplit(':', 1)[1].strip())
        elif 'Comprehensiveness Score' in line:
            metrics['comprehensiveness'] = float(line.rsplit(':', 1)[1].strip())
    return metrics


def normalize_activation(act: torch.Tensor) -> torch.Tensor:
    act = act.detach().float().cpu()
    act = act - act.min()
    denom = act.max().clamp_min(1e-8)
    return act / denom


def tensor_to_heatmap(act: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    act = normalize_activation(act)
    act = act[None, None]
    up = F.interpolate(act, size=size[::-1], mode='nearest').squeeze().numpy()
    return np.clip(up, 0.0, 1.0)


def tensor_to_patch_grid(act: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    act = normalize_activation(act)
    act = act[None, None]
    up = F.interpolate(act, size=size[::-1], mode='nearest').squeeze().numpy()
    return np.clip(up, 0.0, 1.0)


def heat_color(value: np.ndarray) -> np.ndarray:
    """Turbo-like RGB map: blue/green/yellow/red for saliency overlays."""
    x = np.clip(value, 0.0, 1.0)
    anchors = np.array([
        [48, 18, 59],
        [50, 100, 185],
        [35, 180, 170],
        [115, 205, 85],
        [245, 220, 55],
        [235, 120, 35],
        [150, 25, 30],
    ], dtype=np.float32)
    pos = x * (len(anchors) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(anchors) - 1)
    t = (pos - lo)[..., None]
    rgb = anchors[lo] * (1.0 - t) + anchors[hi] * t
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_overlay_and_crop(image_path: str, activation: torch.Tensor, out_base: Path, half_size: int = 36) -> tuple[str, str, str, str]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert('RGB').resize((224, 224), Image.BICUBIC)
    heat = tensor_to_heatmap(activation, image.size)
    heat_img = Image.fromarray(heat_color(heat))
    image_arr = np.asarray(image).astype(np.float32)
    heat_arr = np.asarray(heat_img).astype(np.float32)
    alpha = np.clip((heat - 0.08) / 0.92, 0.0, 1.0) ** 0.72
    alpha = (alpha * 0.64)[..., None]
    overlay_arr = image_arr * (1.0 - alpha) + heat_arr * alpha
    overlay = Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8))

    flat_idx = int(np.argmax(heat))
    y, x = divmod(flat_idx, heat.shape[1])
    left = max(0, x - half_size)
    top = max(0, y - half_size)
    right = min(image.size[0], x + half_size)
    bottom = min(image.size[1], y + half_size)

    draw = ImageDraw.Draw(overlay)
    draw.rectangle([left, top, right, bottom], outline=(25, 28, 24), width=5)
    draw.rectangle([left, top, right, bottom], outline=(255, 245, 180), width=3)

    crop = image.crop((left, top, right, bottom)).resize((160, 160), Image.BICUBIC)
    activation_grid = tensor_to_patch_grid(activation, image.size)
    activation_img = Image.fromarray(heat_color(activation_grid))

    overlay_path = out_base.with_name(out_base.name + '_overlay.jpg')
    original_path = out_base.with_name(out_base.name + '_original.jpg')
    crop_path = out_base.with_name(out_base.name + '_crop.jpg')
    activation_path = out_base.with_name(out_base.name + '_activation.jpg')
    overlay.save(overlay_path, quality=92)
    image.save(original_path, quality=92)
    crop.save(crop_path, quality=92)
    activation_img.save(activation_path, quality=92)
    return overlay_path.as_posix(), original_path.as_posix(), crop_path.as_posix(), activation_path.as_posix()


def update_topk(store: dict[tuple[int, int], list[tuple[float, str, torch.Tensor]]], key: tuple[int, int], item: tuple[float, str, torch.Tensor], k: int) -> None:
    bucket = store[key]
    bucket.append(item)
    bucket.sort(key=lambda x: x[0], reverse=True)
    if len(bucket) > k:
        del bucket[k:]


def scan_top_activations(model: PNP, loader: DataLoader, device: torch.device, topk: int) -> dict[tuple[int, int], list[tuple[float, str, torch.Tensor]]]:
    n_proto = model.n_prototypes
    top: dict[tuple[int, int], list[tuple[float, str, torch.Tensor]]] = defaultdict(list)
    with torch.inference_mode():
        for images, labels, paths in tqdm(loader, desc='Scanning prototype activations'):
            images = images.to(device)
            labels = labels.to(device)
            raw_maps, _ = model.get_attn_maps(images, labels)
            # raw_maps: [B, K, H, W] for each image's ground-truth class prototypes.
            scores = raw_maps.flatten(2).max(dim=-1).values
            for b in range(images.size(0)):
                cls = int(labels[b].item())
                for part_idx in range(n_proto):
                    update_topk(
                        top,
                        (cls, part_idx),
                        (float(scores[b, part_idx].item()), paths[b], raw_maps[b, part_idx].detach().cpu()),
                        topk,
                    )
    return top


def scan_owner_activations(model: PNP, loader: DataLoader, device: torch.device) -> dict[tuple[int, int], list[tuple[float, str, torch.Tensor]]]:
    n_proto = model.n_prototypes
    store: dict[tuple[int, int], list[tuple[float, str, torch.Tensor]]] = defaultdict(list)
    with torch.inference_mode():
        for images, labels, paths in tqdm(loader, desc='Scanning all owner-class prototype activations'):
            images = images.to(device)
            labels = labels.to(device)
            raw_maps, _ = model.get_attn_maps(images, labels)
            scores = raw_maps.flatten(2).max(dim=-1).values
            for b in range(images.size(0)):
                cls = int(labels[b].item())
                for part_idx in range(n_proto):
                    store[(cls, part_idx)].append((
                        float(scores[b, part_idx].item()),
                        paths[b],
                        raw_maps[b, part_idx].detach().cpu(),
                    ))
    for values in store.values():
        values.sort(key=lambda item: item[0], reverse=True)
    return store


def get_clip_cache(device: torch.device, prompts: list[str]):
    global _CLIP_CACHE
    if _CLIP_CACHE is not None:
        return _CLIP_CACHE
    try:
        import clip  # type: ignore
    except Exception as exc:
        _CLIP_CACHE = {'available': False, 'error': f'CLIP import failed: {exc}'}
        return _CLIP_CACHE

    clip_model, preprocess = clip.load('ViT-B/32', device=device.type)
    clip_model.eval()
    text_tokens = clip.tokenize([f'a photo of a {p}' for p in prompts]).to(device)
    with torch.inference_mode():
        text_features = clip_model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    _CLIP_CACHE = {
        'available': True,
        'model': clip_model,
        'preprocess': preprocess,
        'text_features': text_features,
        'prompts': prompts,
    }
    return _CLIP_CACHE


def try_clip_label(crop_paths: list[str], device: torch.device, prompts: list[str]) -> tuple[str, float, list[str]]:
    if not crop_paths:
        return 'no evidence', 0.0, []

    cache = get_clip_cache(device, prompts)
    if not cache.get('available'):
        return 'clip unavailable', 0.0, [cache.get('error', 'CLIP unavailable')]

    clip_model = cache['model']
    preprocess = cache['preprocess']
    text_features = cache['text_features']
    image_tensors = [preprocess(Image.open(path).convert('RGB')) for path in crop_paths]
    image_batch = torch.stack(image_tensors).to(device)
    with torch.inference_mode():
        image_features = clip_model.encode_image(image_batch).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        mean_image = image_features.mean(dim=0, keepdim=True)
        mean_image = mean_image / mean_image.norm(dim=-1, keepdim=True)
        sims = (mean_image @ text_features.T).squeeze(0)
        values, indices = sims.topk(k=min(3, len(prompts)))
    candidates = [prompts[i] for i in indices.tolist()]
    return candidates[0], float(values[0].item()), candidates


def build_prediction_examples(
    model: PNP,
    dataset: ImageFolderWithPath,
    class_names: list[str],
    prototypes: list[dict[str, Any]],
    report_root: Path,
    device: torch.device,
    count: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    visual_prototypes: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    output_dir = report_root / 'examples'
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(count, len(dataset))
    indices = random.Random(seed).sample(range(len(dataset)), sample_count)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    proto_lookup = {item['prototype_id']: item for item in prototypes}
    n_proto = model.n_prototypes
    sa_weights = F.softmax(model.classifier.weights.detach().cpu(), dim=-1) * n_proto
    examples = []
    with torch.inference_mode():
        for images, labels, paths in loader:
            images = images.to(device)
            outputs = model(images)
            probs = outputs['class_logits'].softmax(dim=-1)
            scores, preds = probs.max(dim=-1)
            image_proto_logits = outputs['image_prototype_logits'][:, :-1, :].detach().cpu()
            patch_proto_logits = outputs['patch_prototype_logits'][:, :, :-1, :].detach().cpu()
            B, n_patches, n_classes, _ = patch_proto_logits.shape
            H = W = int(math.sqrt(n_patches))
            patch_proto_maps = rearrange(
                patch_proto_logits,
                'B (H W) C K -> B C K H W',
                H=H,
                W=W,
            )
            for b in range(images.size(0)):
                if len(examples) >= count:
                    return examples
                src = Image.open(paths[b]).convert('RGB').resize((224, 224), Image.BICUBIC)
                pred = int(preds[b].item())
                true = int(labels[b].item())
                example_idx = len(examples)
                example_dir = output_dir / f'prediction_{example_idx:03d}_evidence'
                example_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f'prediction_{example_idx:03d}.jpg'
                src.save(out_path, quality=92)

                all_scores = []
                for cls in range(n_classes):
                    for part_idx in range(n_proto):
                        flat_idx = cls * n_proto + part_idx
                        prototype_id = f'proto_{flat_idx:04d}'
                        activation_score = float(image_proto_logits[b, cls, part_idx].item())
                        classifier_weight = float(sa_weights[cls, part_idx].item())
                        owner_logit_contribution = activation_score * classifier_weight / float(model.temperature)
                        proto_meta = proto_lookup.get(prototype_id, {})
                        all_scores.append({
                            'prototype_id': prototype_id,
                            'class_idx': cls + 1,
                            'owner_class': class_names[cls] if cls < len(class_names) else f'class_{cls:03d}',
                            'part_idx': part_idx,
                            'primary_label': proto_meta.get('primary_label', ''),
                            'annotation_label': proto_meta.get('annotation_label', ''),
                            'annotation_confidence': proto_meta.get('annotation_confidence', None),
                            'clip_label': proto_meta.get('clip_label', ''),
                            'activation_score': activation_score,
                            'classifier_weight': classifier_weight,
                            'owner_logit_contribution': owner_logit_contribution,
                            'contributes_to_prediction': cls == pred,
                        })

                pred_class_ids = {f'proto_{pred * n_proto + part_idx:04d}' for part_idx in range(n_proto)}
                top_by_activation = sorted(all_scores, key=lambda item: item['activation_score'], reverse=True)
                visual_ids = set(pred_class_ids)
                for item in top_by_activation:
                    visual_ids.add(item['prototype_id'])
                    if len(visual_ids) >= max(visual_prototypes, len(pred_class_ids)):
                        break

                visual_assets = {}
                for item in all_scores:
                    if item['prototype_id'] not in visual_ids:
                        continue
                    cls = int(item['class_idx']) - 1
                    part_idx = int(item['part_idx'])
                    activation = patch_proto_maps[b, cls, part_idx]
                    out_base = example_dir / f"{item['prototype_id']}_{safe_name(item['owner_class'])}"
                    overlay, original, crop, activation_path = save_overlay_and_crop(paths[b], activation, out_base)
                    box = activation_box_from_map(activation)
                    visual_assets[item['prototype_id']] = {
                        'overlay': Path(overlay).relative_to(report_root).as_posix(),
                        'original': Path(original).relative_to(report_root).as_posix(),
                        'crop': Path(crop).relative_to(report_root).as_posix(),
                        'activation': Path(activation_path).relative_to(report_root).as_posix(),
                        'box': box,
                    }

                for item in all_scores:
                    assets = visual_assets.get(item['prototype_id'])
                    if assets:
                        item.update(assets)

                predicted_class_evidence = sorted(
                    [item for item in all_scores if item['contributes_to_prediction']],
                    key=lambda item: item['owner_logit_contribution'],
                    reverse=True,
                )
                top_activation_evidence = top_by_activation[:visual_prototypes]
                examples.append({
                    'image': out_path.relative_to(report_root).as_posix(),
                    'image_path': paths[b],
                    'predicted_class': class_names[pred],
                    'true_class': class_names[true],
                    'score': float(scores[b].item()),
                    'predicted_class_idx': pred + 1,
                    'true_class_idx': true + 1,
                    'is_correct': pred == true,
                    'prototype_evidence': all_scores,
                    'predicted_class_evidence': predicted_class_evidence,
                    'top_activation_evidence': top_activation_evidence,
                    'summary': 'Random held-out CUB image with the model top softmax prediction.',
                })
    return examples


def write_report_data(report_root: Path, payload: dict[str, Any]) -> None:
    data_dir = report_root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / 'report_data.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (data_dir / 'report_data.js').write_text('window.REPORT_DATA = ' + json.dumps(payload) + ';\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default='./log-dir-paper-run/ckpt.pth')
    parser.add_argument('--dataset-root', default='./dataset-root')
    parser.add_argument('--report-root', default='./interpretability_report')
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--vote-percentile', type=float, default=75.0)
    parser.add_argument('--min-vote-items', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prediction-examples', type=int, default=12)
    parser.add_argument('--prediction-visual-prototypes', type=int, default=20)
    parser.add_argument('--skip-clip', action='store_true')
    parser.add_argument('--clear', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    ckpt_path = Path(args.ckpt_path)
    dataset_root = Path(args.dataset_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for name in ('data', 'examples', 'prototypes'):
            target = report_root / name
            if target.exists():
                shutil.rmtree(target)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info('Using device: %s', device)
    model, hparams, ckpt = load_checkpoint_model(ckpt_path, device)
    class_names = load_classes(dataset_root)
    part_metadata = load_part_metadata(dataset_root)

    split_dir = 'test_cropped' if args.split == 'test' else 'train_cropped_augmented'
    image_root = dataset_root / 'cub200_cropped' / split_dir
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    dataset = ImageFolderWithPath(image_root.as_posix(), transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    all_owner_activations = scan_owner_activations(model, loader, device)
    top = {
        key: values[:args.topk]
        for key, values in all_owner_activations.items()
    }
    prototypes = []
    n_classes = model.n_classes
    n_proto = model.n_prototypes
    proto_root = report_root / 'prototypes'

    for cls in tqdm(range(n_classes), desc='Writing prototype assets'):
        class_name = class_names[cls] if cls < len(class_names) else f'class_{cls:03d}'
        class_dir = proto_root / f'class_{cls + 1:03d}_{safe_name(class_name)}'
        for part_idx in range(n_proto):
            flat_idx = cls * n_proto + part_idx
            prototype_id = f'proto_{flat_idx:04d}'
            items = top.get((cls, part_idx), [])
            examples = []
            crop_paths = []
            for rank, (score, image_path, activation) in enumerate(items, start=1):
                out_base = class_dir / f'{prototype_id}_top_{rank:02d}'
                overlay, original, crop, activation_path = save_overlay_and_crop(image_path, activation, out_base)
                crop_paths.append(crop)
                examples.append({
                    'rank': rank,
                    'activation_score': score,
                    'image_path': image_path,
                    'overlay': Path(overlay).relative_to(report_root).as_posix(),
                    'original': Path(original).relative_to(report_root).as_posix(),
                    'crop': Path(crop).relative_to(report_root).as_posix(),
                    'activation': Path(activation_path).relative_to(report_root).as_posix(),
                })

            vote_items = all_owner_activations.get((cls, part_idx), [])
            annotation_label, annotation_confidence, annotation_evidence, annotation_votes, part_vote_process = vote_part_label_gaussian(
                vote_items,
                part_metadata,
                dataset_root,
                image_root,
                vote_percentile=args.vote_percentile,
                min_vote_items=args.min_vote_items,
            )
            if args.skip_clip:
                clip_label, clip_score, clip_candidates = 'not labeled', 0.0, []
            else:
                clip_label, clip_score, clip_candidates = try_clip_label(crop_paths, device, PART_PROMPTS)

            primary_label = annotation_label if annotation_label != 'part unclear' else clip_label
            prototypes.append({
                'prototype_id': prototype_id,
                'class_idx': cls + 1,
                'class_name': class_name,
                'part_idx': part_idx,
                'primary_label': primary_label,
                'annotation_label': annotation_label,
                'annotation_confidence': annotation_confidence,
                'annotation_evidence': annotation_evidence,
                'annotation_votes': annotation_votes,
                'part_vote_process': part_vote_process,
                'vote_method': part_vote_process.get('method', ''),
                'vote_percentile': args.vote_percentile,
                'sigma_rule': part_vote_process.get('sigma_rule', ''),
                'clip_label': clip_label,
                'clip_score': clip_score,
                'clip_candidates': clip_candidates,
                'thumbnail': examples[0]['overlay'] if examples else '',
                'thumbnail_overlay': examples[0]['overlay'] if examples else '',
                'thumbnail_original': examples[0]['original'] if examples else '',
                'thumbnail_activation': examples[0]['activation'] if examples else '',
                'examples': examples,
            })

    # Recreate a loader because CLIP may have consumed GPU memory but model remains available.
    predictions = build_prediction_examples(
        model,
        dataset,
        class_names,
        prototypes,
        report_root,
        device,
        args.prediction_examples,
        args.batch_size,
        args.num_workers,
        seed=42,
        visual_prototypes=args.prediction_visual_prototypes,
    )

    metrics = parse_metrics(ckpt_path.parent / 'train.log')
    model_summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint_path': ckpt_path.as_posix(),
        'checkpoint_name': ckpt_path.name,
        'dataset_root': dataset_root.as_posix(),
        'split': args.split,
        'backbone': getattr(hparams, 'backbone', 'unknown'),
        'num_classes': n_classes,
        'num_prototypes_per_class': n_proto,
        'num_foreground_prototypes': n_classes * n_proto,
        'prototype_tensor_shape': list(model.prototypes.shape),
        'images_scanned': len(dataset),
        'topk_per_prototype': args.topk,
        'vote_method': 'owner-class percentile gaussian proximity voting',
        'vote_percentile': args.vote_percentile,
        'min_vote_items': args.min_vote_items,
        'sigma_rule': 'activation box diagonal / 2',
    }
    payload = {
        'model_summary': model_summary,
        'metrics': metrics,
        'prototypes': prototypes,
        'predictions': predictions,
    }
    write_report_data(report_root, payload)
    LOGGER.info('Report assets written to %s', report_root)
    LOGGER.info('Copy data/, prototypes/, and examples/ next to the local index.html viewer.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
