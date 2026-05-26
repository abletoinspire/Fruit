# python train.py --train-img-dir .\edited_dataset\train\images --train-ann .\edited_dataset\train\annotations.json --val-img-dir .\edited_dataset\val\images --val-ann .\edited_dataset\val\annotations.json --out-dir .\runs_multihead --cache-dir .\mask_cache --precache --img-size 512 --epochs 80 --batch-size 8 --cm-every 1 --min-obj-area 200
import os
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


try:
    from torch.amp import autocast
    AUTocast_OK = True
except Exception:
    from torch.cuda.amp import autocast
    AUTocast_OK = True

try:
    from torch.amp import GradScaler
    SCALER_OK = True
except Exception:
    from torch.cuda.amp import GradScaler
    SCALER_OK = True


try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False


@dataclass
class Config:
    train_img_dir: str
    train_ann: str
    val_img_dir: str
    val_ann: str
    out_dir: str

    img_size: int = 512
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 80
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2

    base_c: int = 48
    use_amp: bool = True
    seed: int = 42

    ignore_index: int = 255


    mask_cache_dir: str = ""


    enable_tf32: bool = True
    use_channels_last: bool = True
    empty_cache_each_epoch: bool = True


    w_fruit: float = 1.0


    w_type_px: float = 0.20
    w_type_obj: float = 0.80
    w_qual_px: float = 0.20
    w_qual_obj: float = 0.80

    dice_w: float = 1.0


    obj_loss_min_area: int = 200


    use_class_weights: bool = True
    weight_mode: str = "median"
    label_smoothing: float = 0.05


    grad_clip: float = 1.0
    skip_nonfinite: bool = True


    hflip_p: float = 0.5
    brightness: float = 0.12
    contrast: float = 0.12
    blur_p: float = 0.10
    noise_p: float = 0.10
    jpeg_p: float = 0.12
    jpeg_qmin: int = 55
    jpeg_qmax: int = 95
    gamma_p: float = 0.10
    gamma_min: float = 0.85
    gamma_max: float = 1.15


    cm_every: int = 1
    min_obj_area: int = 200


    vis_every: int = 10
    max_vis: int = 3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device_type(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def cuda_mem_mb() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserv = torch.cuda.memory_reserved() / (1024 ** 2)
    return float(alloc), float(reserv)


def bincount_cm(gt: np.ndarray, pr: np.ndarray, n: int) -> np.ndarray:
    idx = gt.astype(np.int64) * n + pr.astype(np.int64)
    cm = np.bincount(idx, minlength=n * n).reshape(n, n)
    return cm


def save_cm_png(cm: np.ndarray, class_names: List[str], title: str, out_path: str):
    if not MPL_OK:
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("GT")
    plt.xlabel("Pred")

    vmax = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if vmax > 0:
                plt.text(j, i, f"{val}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def parse_cat_name(name: str) -> Tuple[str, str]:
    n = (name or "").lower().strip()
    parts = n.split("_")
    if len(parts) >= 2:
        quality = parts[0]
        fruit = parts[-1]
        return quality, fruit

    if "fresh" in n:
        q = "fresh"
    elif "rotten" in n:
        q = "rotten"
    else:
        q = "unknown"

    for f in ("apple", "banana", "orange"):
        if f in n:
            return q, f
    return q, "unknown"


class CocoIndex:
    def __init__(self, ann_path: str):
        coco = load_json(ann_path)
        self.coco = coco
        self.images = coco.get("images", [])
        self.annotations = coco.get("annotations", [])
        self.categories = coco.get("categories", [])

        self.imgid_to_img = {im["id"]: im for im in self.images}
        self.imgid_to_anns: Dict[int, List[dict]] = {}
        for a in self.annotations:
            self.imgid_to_anns.setdefault(a["image_id"], []).append(a)

        self.catid_to_name = {c["id"]: c.get("name", "") for c in self.categories}


        self.type_names = ["apple", "banana", "orange"]
        self.type_to_id = {t: i for i, t in enumerate(self.type_names)}

        self.quality_names = ["fresh", "rotten"]
        self.quality_to_id = {"fresh": 0, "rotten": 1}

        self.catid_to_type_quality: Dict[int, Tuple[int, int]] = {}
        for cid, nm in self.catid_to_name.items():
            q, fruit = parse_cat_name(nm)
            if fruit not in self.type_to_id:
                continue
            if q not in self.quality_to_id:
                continue
            self.catid_to_type_quality[int(cid)] = (self.type_to_id[fruit], self.quality_to_id[q])

    @property
    def num_types(self) -> int:
        return len(self.type_names)

    @property
    def num_qualities(self) -> int:
        return len(self.quality_names)


def compute_obj_class_counts_from_train(idx: CocoIndex) -> Tuple[np.ndarray, np.ndarray]:
    t_counts = np.zeros((idx.num_types,), dtype=np.int64)
    q_counts = np.zeros((idx.num_qualities,), dtype=np.int64)
    for a in idx.annotations:
        if a.get("iscrowd", 0) == 1:
            continue
        cid = int(a.get("category_id", -1))
        if cid not in idx.catid_to_type_quality:
            continue
        t_id, q_id = idx.catid_to_type_quality[cid]
        if 0 <= t_id < idx.num_types:
            t_counts[t_id] += 1
        if 0 <= q_id < idx.num_qualities:
            q_counts[q_id] += 1
    return t_counts, q_counts


def make_class_weights(counts: np.ndarray, mode: str) -> np.ndarray:
    c = counts.astype(np.float64).copy()
    c[c <= 0] = 1.0
    mode = (mode or "none").lower().strip()

    if mode == "none":
        w = np.ones_like(c)
    elif mode == "inv":
        w = 1.0 / c
    elif mode == "inv_sqrt":
        w = 1.0 / np.sqrt(c)
    else:

        med = float(np.median(c))
        w = med / c


    w = w / float(np.mean(w))
    return w.astype(np.float32)


def polygons_to_mask(segmentation: List[List[float]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def build_targets(
    anns: List[dict],
    idx: CocoIndex,
    h: int,
    w: int,
    ignore_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fruit_mask = np.zeros((h, w), dtype=np.uint8)
    type_map = np.full((h, w), ignore_index, dtype=np.uint8)
    qual_map = np.full((h, w), ignore_index, dtype=np.uint8)

    def ann_area(a: dict) -> float:
        if "area" in a and a["area"] is not None:
            return float(a["area"])
        x, y, bw, bh = a.get("bbox", [0, 0, 0, 0])
        return float(bw) * float(bh)


    anns_sorted = sorted(anns, key=ann_area, reverse=True)

    for a in anns_sorted:
        if a.get("iscrowd", 0) == 1:
            continue
        cid = int(a.get("category_id", -1))
        if cid not in idx.catid_to_type_quality:
            continue
        seg = a.get("segmentation", [])
        if not seg:
            continue

        m = polygons_to_mask(seg, h, w)
        if m.sum() == 0:
            continue

        t_id, q_id = idx.catid_to_type_quality[cid]
        fruit_mask[m == 1] = 1
        type_map[m == 1] = t_id
        qual_map[m == 1] = q_id

    return fruit_mask, type_map, qual_map


def letterbox(img: np.ndarray, out_size: int, is_mask: bool = False, pad_value: int = 0):
    h, w = img.shape[:2]
    scale = out_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    pad_y = out_size - nh
    pad_x = out_size - nw
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left

    if img.ndim == 2:
        out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=pad_value)
    else:
        out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    return out


def _jpeg_recompress_rgb(img_rgb: np.ndarray, q: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), encode_param)
    if not ok:
        return img_rgb
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        return img_rgb
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def _gamma_rgb(img_rgb: np.ndarray, gamma: float) -> np.ndarray:
    g = float(max(1e-3, gamma))
    x = img_rgb.astype(np.float32) / 255.0
    x = np.power(x, g)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def aug_train(img_rgb: np.ndarray, fruit: np.ndarray, tmap: np.ndarray, qmap: np.ndarray, cfg: Config):
    if random.random() < cfg.hflip_p:
        img_rgb = np.ascontiguousarray(np.fliplr(img_rgb))
        fruit = np.ascontiguousarray(np.fliplr(fruit))
        tmap = np.ascontiguousarray(np.fliplr(tmap))
        qmap = np.ascontiguousarray(np.fliplr(qmap))

    if cfg.brightness > 0 or cfg.contrast > 0:
        b = random.uniform(-cfg.brightness, cfg.brightness) * 255.0
        c = random.uniform(1.0 - cfg.contrast, 1.0 + cfg.contrast)
        img_f = img_rgb.astype(np.float32) * c + b
        img_rgb = np.clip(img_f, 0, 255).astype(np.uint8)

    if random.random() < cfg.gamma_p:
        g = random.uniform(cfg.gamma_min, cfg.gamma_max)
        img_rgb = _gamma_rgb(img_rgb, g)

    if random.random() < cfg.blur_p:
        k = random.choice([3, 5])
        img_rgb = cv2.GaussianBlur(img_rgb, (k, k), 0)

    if random.random() < cfg.noise_p:
        sigma = random.uniform(2.0, 8.0)
        noise = np.random.normal(0, sigma, img_rgb.shape).astype(np.float32)
        img_rgb = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < cfg.jpeg_p:
        q = random.randint(int(cfg.jpeg_qmin), int(cfg.jpeg_qmax))
        img_rgb = _jpeg_recompress_rgb(img_rgb, q)

    return img_rgb, fruit, tmap, qmap


class MultiHeadCocoDataset(Dataset):
    def __init__(self, img_dir: str, ann_path: str, cfg: Config, is_train: bool, split_name: str):
        self.img_dir = img_dir
        self.cfg = cfg
        self.is_train = is_train
        self.split_name = split_name

        self.idx = CocoIndex(ann_path)
        self.images = list(self.idx.imgid_to_img.values())

        self.cache_dir = (cfg.mask_cache_dir or "").strip()
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.split_name)
            ensure_dir(self.cache_dir)

    def __len__(self):
        return len(self.images)

    def _cache_path(self, file_name: str) -> str:
        base = os.path.splitext(os.path.basename(file_name))[0]
        return os.path.join(self.cache_dir, base + ".npz")

    def _load_or_build_targets(self, img_id: int, file_name: str, h: int, w: int):
        if self.cache_dir:
            cpath = self._cache_path(file_name)
            if os.path.exists(cpath):
                data = np.load(cpath)
                fruit = data["fruit"].astype(np.uint8)
                tmap = data["type"].astype(np.uint8)
                qmap = data["qual"].astype(np.uint8)
                return fruit, tmap, qmap

        anns = self.idx.imgid_to_anns.get(img_id, [])
        fruit, tmap, qmap = build_targets(anns, self.idx, h, w, self.cfg.ignore_index)

        if self.cache_dir:
            cpath = self._cache_path(file_name)
            tmp_path = cpath + ".tmp.npz"
            np.savez_compressed(tmp_path, fruit=fruit, type=tmap, qual=qmap)
            os.replace(tmp_path, cpath)

        return fruit, tmap, qmap

    def __getitem__(self, i: int):
        info = self.images[i]
        img_id = info["id"]
        file_name = info["file_name"]
        path = os.path.join(self.img_dir, file_name)

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        fruit, tmap, qmap = self._load_or_build_targets(img_id, file_name, h, w)

        if self.is_train:
            rgb, fruit, tmap, qmap = aug_train(rgb, fruit, tmap, qmap, self.cfg)

        rgb_lb = letterbox(rgb, self.cfg.img_size, is_mask=False, pad_value=0)
        fruit_lb = letterbox(fruit, self.cfg.img_size, is_mask=True, pad_value=0)
        tmap_lb = letterbox(tmap, self.cfg.img_size, is_mask=True, pad_value=self.cfg.ignore_index)
        qmap_lb = letterbox(qmap, self.cfg.img_size, is_mask=True, pad_value=self.cfg.ignore_index)

        x = torch.from_numpy(rgb_lb).float().permute(2, 0, 1) / 255.0
        fruit_t = torch.from_numpy(fruit_lb).float().unsqueeze(0)
        type_t = torch.from_numpy(tmap_lb).long()
        qual_t = torch.from_numpy(qmap_lb).long()

        return x, fruit_t, type_t, qual_t, file_name


def precache_dataset(ds: MultiHeadCocoDataset):
    if not ds.cache_dir:
        return
    for i in tqdm(range(len(ds)), desc=f"Precache {ds.split_name}"):
        info = ds.images[i]
        img_id = info["id"]
        file_name = info["file_name"]

        img_path = os.path.join(ds.img_dir, file_name)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)
        h, w = bgr.shape[:2]
        _ = ds._load_or_build_targets(img_id, file_name, h, w)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, groups: int = 8):
        super().__init__()
        g = min(groups, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, out_ch),
            ConvGNAct(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SegClassHead(nn.Module):
    def __init__(self, ch: int, out_classes: int, drop_p: float = 0.10):
        super().__init__()
        g = min(8, ch)
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=float(drop_p)),
            nn.Conv2d(ch, out_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetMultiHead(nn.Module):
    def __init__(self, base_c: int, num_types: int, num_qualities: int, head_drop: float = 0.10):
        super().__init__()
        c1 = base_c
        c2 = base_c * 2
        c3 = base_c * 4
        c4 = base_c * 8
        c5 = base_c * 8

        self.inc = DoubleConv(3, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5 + c4, c3)
        self.up2 = Up(c3 + c3, c2)
        self.up3 = Up(c2 + c2, c1)
        self.up4 = Up(c1 + c1, c1)

        self.head_fruit = nn.Conv2d(c1, 1, kernel_size=1)
        self.head_type = SegClassHead(c1, num_types, drop_p=head_drop)
        self.head_quality = SegClassHead(c1, num_qualities, drop_p=head_drop)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.head_fruit(x), self.head_type(x), self.head_quality(x)


def dice_loss_from_logits(logits_fp32: torch.Tensor, target_fp32: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits_fp32)
    prob = prob.contiguous().view(prob.size(0), -1)
    tgt = target_fp32.contiguous().view(target_fp32.size(0), -1)
    inter = (prob * tgt).sum(dim=1)
    union = prob.sum(dim=1) + tgt.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def safe_pixel_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    mask = (target != ignore_index)
    if mask.sum().item() == 0:
        return logits.sum() * 0.0


    x = logits.permute(0, 2, 3, 1)[mask]
    y = target[mask]
    try:
        return F.cross_entropy(x, y, weight=weight, label_smoothing=float(label_smoothing))
    except TypeError:

        return F.cross_entropy(x, y, weight=weight)


def object_ce_loss_from_gt_components(
    logits: torch.Tensor,
    target_map: torch.Tensor,
    fruit_gt: torch.Tensor,
    ignore_index: int,
    num_classes: int,
    min_area: int,
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    device = logits.device
    B = logits.size(0)
    obj_logits = []
    obj_targets = []

    for b in range(B):
        fg = (fruit_gt[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        if fg.sum() == 0:
            continue

        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if nlab <= 1:
            continue

        tmap = target_map[b].detach().cpu().numpy()
        for lbl in range(1, nlab):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue

            m = (labels == lbl)

            m_gt = m & (tmap != int(ignore_index))
            if m_gt.sum() == 0:
                continue

            gt_vals = tmap[m_gt].astype(np.int64)
            gt_cls = int(np.bincount(gt_vals, minlength=num_classes).argmax())


            m_t = torch.from_numpy(m_gt).to(device=device, dtype=torch.bool)

            l = logits[b, :, :, :][:, m_t].mean(dim=1)
            obj_logits.append(l)
            obj_targets.append(gt_cls)

    if len(obj_logits) == 0:
        return logits.sum() * 0.0

    X = torch.stack(obj_logits, dim=0)
    y = torch.tensor(obj_targets, device=device, dtype=torch.long)
    try:
        return F.cross_entropy(X, y, weight=weight, label_smoothing=float(label_smoothing))
    except TypeError:
        return F.cross_entropy(X, y, weight=weight)


@torch.no_grad()
def bin_iou_and_dice(fruit_logits: torch.Tensor, fruit_gt: torch.Tensor, thr: float = 0.5) -> Tuple[float, float]:
    prob = torch.sigmoid(fruit_logits)
    pred = (prob > thr).to(torch.uint8)
    gt = (fruit_gt > 0.5).to(torch.uint8)

    pred_f = pred.view(pred.size(0), -1)
    gt_f = gt.view(gt.size(0), -1)

    inter = (pred_f & gt_f).sum(dim=1).float()
    union = (pred_f | gt_f).sum(dim=1).float()
    sum_ = pred_f.sum(dim=1).float() + gt_f.sum(dim=1).float()

    iou = torch.where(union > 0, inter / (union + 1e-6), torch.ones_like(union))
    dice = torch.where(sum_ > 0, (2 * inter) / (sum_ + 1e-6), torch.ones_like(sum_))
    return float(iou.mean().item()), float(dice.mean().item())


@torch.no_grad()
def masked_acc(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> float:
    pred = torch.argmax(logits, dim=1)
    mask = (target != ignore_index)
    if mask.sum().item() == 0:
        return 1.0
    return float((pred[mask] == target[mask]).float().mean().item())


@torch.no_grad()
def masked_miou(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> float:
    pred = torch.argmax(logits, dim=1)
    mask = (target != ignore_index)
    if mask.sum().item() == 0:
        return 1.0

    ious = []
    for cls in range(num_classes):
        p = (pred == cls) & mask
        t = (target == cls) & mask
        if t.sum().item() == 0:
            continue
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / (union + 1e-6))
    return float(np.mean(ious)) if ious else 0.0


def update_pixel_cm(cm: np.ndarray, gt_t: torch.Tensor, pr_t: torch.Tensor, ignore_index: int, n: int):
    gt = gt_t.detach().cpu().numpy()
    pr = pr_t.detach().cpu().numpy()
    m = (gt != ignore_index)
    if m.sum() == 0:
        return
    cm += bincount_cm(gt[m], pr[m], n)


def update_object_cm(
    cm_type_obj: np.ndarray,
    cm_qual_obj: np.ndarray,
    fruit_gt: torch.Tensor,
    type_gt: torch.Tensor,
    qual_gt: torch.Tensor,
    type_pr: torch.Tensor,
    qual_pr: torch.Tensor,
    ignore_index: int,
    num_types: int,
    num_qualities: int,
    min_area: int,
):
    fg = (fruit_gt.detach().cpu().numpy().squeeze(0) > 0.5).astype(np.uint8)
    if fg.sum() == 0:
        return

    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if nlab <= 1:
        return

    t_gt = type_gt.detach().cpu().numpy()
    q_gt = qual_gt.detach().cpu().numpy()
    t_pr = type_pr.detach().cpu().numpy()
    q_pr = qual_pr.detach().cpu().numpy()

    for lbl in range(1, nlab):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        m = (labels == lbl)
        m_gt = m & (t_gt != ignore_index) & (q_gt != ignore_index)
        if m_gt.sum() == 0:
            continue

        gt_t_vals = t_gt[m_gt]
        gt_q_vals = q_gt[m_gt]
        gt_t = int(np.bincount(gt_t_vals, minlength=num_types).argmax())
        gt_q = int(np.bincount(gt_q_vals, minlength=num_qualities).argmax())

        pr_t_vals = t_pr[m_gt]
        pr_q_vals = q_pr[m_gt]
        pr_t = int(np.bincount(pr_t_vals, minlength=num_types).argmax())
        pr_q = int(np.bincount(pr_q_vals, minlength=num_qualities).argmax())

        cm_type_obj[gt_t, pr_t] += 1
        cm_qual_obj[gt_q, pr_q] += 1


def save_vis_batch(
    out_dir: str,
    epoch: int,
    names: List[str],
    imgs: torch.Tensor,
    fruit_logits: torch.Tensor,
    type_logits: torch.Tensor,
    qual_logits: torch.Tensor,
    max_vis: int = 3,
):
    ensure_dir(out_dir)
    B = imgs.size(0)
    n = min(B, max_vis)

    imgs_np = (imgs[:n].detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    fruit_pr = (torch.sigmoid(fruit_logits[:n]) > 0.5).detach().cpu().numpy().squeeze(1).astype(np.uint8) * 255
    type_pr = torch.argmax(type_logits[:n], dim=1).detach().cpu().numpy().astype(np.uint8)
    qual_pr = torch.argmax(qual_logits[:n], dim=1).detach().cpu().numpy().astype(np.uint8)

    pal_type = np.array([[255, 80, 80], [80, 255, 80], [80, 80, 255]], dtype=np.uint8)
    pal_qual = np.array([[80, 180, 255], [255, 200, 80]], dtype=np.uint8)

    for i in range(n):
        img = imgs_np[i]
        h, w = img.shape[:2]

        tcol = pal_type[type_pr[i].reshape(-1)].reshape(h, w, 3)
        qcol = pal_qual[qual_pr[i].reshape(-1)].reshape(h, w, 3)
        fcol = np.stack([fruit_pr[i]] * 3, axis=2)

        vis = np.zeros((h, w * 4, 3), dtype=np.uint8)
        vis[:, 0:w] = img
        vis[:, w:2*w] = fcol
        vis[:, 2*w:3*w] = tcol
        vis[:, 3*w:4*w] = qcol

        out_path = os.path.join(out_dir, f"epoch_{epoch:03d}_{os.path.splitext(names[i])[0]}.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional["GradScaler"],
    cfg: Config,
    num_types: int,
    num_qualities: int,
    train: bool,
    type_w: Optional[torch.Tensor],
    qual_w: Optional[torch.Tensor],
    cm_pack: Optional[dict] = None,
) -> Dict[str, float]:
    device = cfg.device
    device_type = get_device_type(device)
    model.train(mode=train)

    bce = nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    n_samples = 0

    iou_sum = 0.0
    dice_sum = 0.0
    type_acc_sum = 0.0
    qual_acc_sum = 0.0
    type_miou_sum = 0.0
    qual_miou_sum = 0.0
    n_batches = 0

    use_amp = bool(cfg.use_amp and AUTocast_OK and device_type == "cuda")

    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        pbar = tqdm(loader, desc=("Train" if train else "Val"), leave=False)
        for imgs, fruit_gt, type_gt, qual_gt, _names in pbar:
            imgs = imgs.to(device, non_blocking=True)
            fruit_gt = fruit_gt.to(device, non_blocking=True).float()
            type_gt = type_gt.to(device, non_blocking=True)
            qual_gt = qual_gt.to(device, non_blocking=True)

            if cfg.use_channels_last and device_type == "cuda":
                imgs = imgs.to(memory_format=torch.channels_last)

            if train:
                optimizer.zero_grad(set_to_none=True)


            if use_amp:
                with autocast(device_type=device_type, enabled=True):
                    fruit_logits, type_logits, qual_logits = model(imgs)
                fruit_logits_f = fruit_logits.float()
                type_logits_f = type_logits.float()
                qual_logits_f = qual_logits.float()
            else:
                fruit_logits, type_logits, qual_logits = model(imgs)
                fruit_logits_f = fruit_logits.float()
                type_logits_f = type_logits.float()
                qual_logits_f = qual_logits.float()


            loss_fruit = bce(fruit_logits_f, fruit_gt) + cfg.dice_w * dice_loss_from_logits(fruit_logits_f, fruit_gt)


            loss_type_px = safe_pixel_ce_loss(
                type_logits_f, type_gt, cfg.ignore_index,
                weight=type_w, label_smoothing=cfg.label_smoothing
            ) if cfg.w_type_px > 0 else (type_logits_f.sum() * 0.0)

            loss_qual_px = safe_pixel_ce_loss(
                qual_logits_f, qual_gt, cfg.ignore_index,
                weight=qual_w, label_smoothing=cfg.label_smoothing
            ) if cfg.w_qual_px > 0 else (qual_logits_f.sum() * 0.0)


            loss_type_obj = object_ce_loss_from_gt_components(
                logits=type_logits_f,
                target_map=type_gt,
                fruit_gt=fruit_gt,
                ignore_index=cfg.ignore_index,
                num_classes=num_types,
                min_area=cfg.obj_loss_min_area,
                weight=type_w,
                label_smoothing=cfg.label_smoothing,
            ) if cfg.w_type_obj > 0 else (type_logits_f.sum() * 0.0)

            loss_qual_obj = object_ce_loss_from_gt_components(
                logits=qual_logits_f,
                target_map=qual_gt,
                fruit_gt=fruit_gt,
                ignore_index=cfg.ignore_index,
                num_classes=num_qualities,
                min_area=cfg.obj_loss_min_area,
                weight=qual_w,
                label_smoothing=cfg.label_smoothing,
            ) if cfg.w_qual_obj > 0 else (qual_logits_f.sum() * 0.0)

            loss_type = cfg.w_type_px * loss_type_px + cfg.w_type_obj * loss_type_obj
            loss_qual = cfg.w_qual_px * loss_qual_px + cfg.w_qual_obj * loss_qual_obj

            loss = cfg.w_fruit * loss_fruit + loss_type + loss_qual

            if cfg.skip_nonfinite and (not torch.isfinite(loss).all()):
                if train and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="NaN/Inf_skip")
                continue

            if train:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if cfg.grad_clip and cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.grad_clip and cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

            bs = imgs.size(0)
            loss_sum += float(loss.item()) * bs
            n_samples += bs

            iou, dice = bin_iou_and_dice(fruit_logits, fruit_gt, thr=0.5)
            type_acc = masked_acc(type_logits, type_gt, cfg.ignore_index)
            qual_acc = masked_acc(qual_logits, qual_gt, cfg.ignore_index)
            type_miou = masked_miou(type_logits, type_gt, num_types, cfg.ignore_index)
            qual_miou = masked_miou(qual_logits, qual_gt, num_qualities, cfg.ignore_index)

            iou_sum += iou
            dice_sum += dice
            type_acc_sum += type_acc
            qual_acc_sum += qual_acc
            type_miou_sum += type_miou
            qual_miou_sum += qual_miou
            n_batches += 1


            if (not train) and (cm_pack is not None):
                type_pr = torch.argmax(type_logits, dim=1)
                qual_pr = torch.argmax(qual_logits, dim=1)

                for b in range(bs):
                    update_pixel_cm(cm_pack["cm_type_px"], type_gt[b], type_pr[b], cfg.ignore_index, num_types)
                    update_pixel_cm(cm_pack["cm_qual_px"], qual_gt[b], qual_pr[b], cfg.ignore_index, num_qualities)

                    update_object_cm(
                        cm_type_obj=cm_pack["cm_type_obj"],
                        cm_qual_obj=cm_pack["cm_qual_obj"],
                        fruit_gt=fruit_gt[b],
                        type_gt=type_gt[b],
                        qual_gt=qual_gt[b],
                        type_pr=type_pr[b],
                        qual_pr=qual_pr[b],
                        ignore_index=cfg.ignore_index,
                        num_types=num_types,
                        num_qualities=num_qualities,
                        min_area=cfg.min_obj_area,
                    )

            pbar.set_postfix(loss=float(loss.item()), fruit_iou=iou, qual_acc=qual_acc)

    return {
        "loss": loss_sum / max(1, n_samples),
        "fruit_iou": iou_sum / max(1, n_batches),
        "fruit_dice": dice_sum / max(1, n_batches),
        "type_acc": type_acc_sum / max(1, n_batches),
        "quality_acc": qual_acc_sum / max(1, n_batches),
        "type_miou": type_miou_sum / max(1, n_batches),
        "quality_miou": qual_miou_sum / max(1, n_batches),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-img-dir", required=True)
    ap.add_argument("--train-ann", required=True)
    ap.add_argument("--val-img-dir", required=True)
    ap.add_argument("--val-ann", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--base-c", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--vis-every", type=int, default=10)

    ap.add_argument("--cache-dir", default="", help="Folder for cached targets (.npz). If empty -> no cache.")
    ap.add_argument("--precache", action="store_true")

    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--no-channels-last", action="store_true")
    ap.add_argument("--no-empty-cache", action="store_true")

    ap.add_argument("--w-fruit", type=float, default=1.0)
    ap.add_argument("--dice-w", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)


    ap.add_argument("--w-type-px", type=float, default=0.20)
    ap.add_argument("--w-type-obj", type=float, default=0.80)
    ap.add_argument("--w-qual-px", type=float, default=0.20)
    ap.add_argument("--w-qual-obj", type=float, default=0.80)
    ap.add_argument("--obj-loss-min-area", type=int, default=200)

    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--weight-mode", type=str, default="median", choices=["none", "inv", "inv_sqrt", "median"])
    ap.add_argument("--label-smoothing", type=float, default=0.05)

    ap.add_argument("--cm-every", type=int, default=1, help="Save confusion matrices every N epochs (0 disables).")
    ap.add_argument("--min-obj-area", type=int, default=200)

    args = ap.parse_args()

    cfg = Config(
        train_img_dir=args.train_img_dir,
        train_ann=args.train_ann,
        val_img_dir=args.val_img_dir,
        val_ann=args.val_ann,
        out_dir=args.out_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        base_c=args.base_c,
        use_amp=not args.no_amp,
        seed=args.seed,
        vis_every=args.vis_every,
        mask_cache_dir=args.cache_dir,
        enable_tf32=not args.no_tf32,
        use_channels_last=not args.no_channels_last,
        empty_cache_each_epoch=not args.no_empty_cache,
        w_fruit=args.w_fruit,
        dice_w=args.dice_w,
        grad_clip=args.grad_clip,
        cm_every=args.cm_every,
        min_obj_area=args.min_obj_area,
        w_type_px=args.w_type_px,
        w_type_obj=args.w_type_obj,
        w_qual_px=args.w_qual_px,
        w_qual_obj=args.w_qual_obj,
        obj_loss_min_area=args.obj_loss_min_area,
        use_class_weights=(not args.no_class_weights),
        weight_mode=args.weight_mode,
        label_smoothing=float(args.label_smoothing),
    )

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "weights"))
    ensure_dir(os.path.join(cfg.out_dir, "vis"))
    ensure_dir(os.path.join(cfg.out_dir, "cms"))

    device_type = get_device_type(cfg.device)


    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True
        if cfg.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    train_ds = MultiHeadCocoDataset(cfg.train_img_dir, cfg.train_ann, cfg, is_train=True, split_name="train")
    val_ds = MultiHeadCocoDataset(cfg.val_img_dir, cfg.val_ann, cfg, is_train=False, split_name="val")


    if train_ds.idx.type_names != ["apple", "banana", "orange"]:
        raise ValueError(f"Unexpected type_names in train: {train_ds.idx.type_names}")
    if val_ds.idx.type_names != ["apple", "banana", "orange"]:
        raise ValueError(f"Unexpected type_names in val: {val_ds.idx.type_names}")
    if train_ds.idx.quality_names != ["fresh", "rotten"] or val_ds.idx.quality_names != ["fresh", "rotten"]:
        raise ValueError("Unexpected quality_names.")

    num_types = train_ds.idx.num_types
    num_qualities = train_ds.idx.num_qualities

    if cfg.mask_cache_dir and args.precache:
        print(f"[PRECACHE] cache_dir={cfg.mask_cache_dir}")
        precache_dataset(train_ds)
        precache_dataset(val_ds)
        print("[PRECACHE] готово.")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


    type_w_t = None
    qual_w_t = None
    t_counts, q_counts = compute_obj_class_counts_from_train(train_ds.idx)

    if cfg.use_class_weights and cfg.weight_mode != "none":
        tw = make_class_weights(t_counts, cfg.weight_mode)
        qw = make_class_weights(q_counts, cfg.weight_mode)
        type_w_t = torch.tensor(tw, device=cfg.device, dtype=torch.float32)
        qual_w_t = torch.tensor(qw, device=cfg.device, dtype=torch.float32)

    model = UNetMultiHead(cfg.base_c, num_types=num_types, num_qualities=num_qualities, head_drop=0.10).to(cfg.device)

    if device_type == "cuda" and cfg.use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_lambda(epoch0: int):
        if epoch0 < cfg.warmup_epochs:
            return float(epoch0 + 1) / float(max(1, cfg.warmup_epochs))
        t = (epoch0 - cfg.warmup_epochs) / float(max(1, cfg.epochs - cfg.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_amp = bool(cfg.use_amp and device_type == "cuda")
    scaler = GradScaler() if use_amp else None

    print(f"Device: {cfg.device} | AMP: {use_amp} | TF32: {cfg.enable_tf32 and device_type=='cuda'} | channels_last: {cfg.use_channels_last and device_type=='cuda'}")
    if cfg.mask_cache_dir:
        print(f"Mask cache: {cfg.mask_cache_dir} (train/val subfolders)")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Types: {train_ds.idx.type_names} (num={num_types}) | obj_counts={t_counts.tolist()}")
    print(f"Qualities: {train_ds.idx.quality_names} (num={num_qualities}) | obj_counts={q_counts.tolist()}")
    print(f"Cls loss: type(px={cfg.w_type_px}, obj={cfg.w_type_obj}) qual(px={cfg.w_qual_px}, obj={cfg.w_qual_obj}) "
          f"| class_weights={cfg.use_class_weights} mode={cfg.weight_mode} | label_smoothing={cfg.label_smoothing}")
    if type_w_t is not None:
        print(f"Type weights:  {type_w_t.detach().cpu().numpy().round(3).tolist()}")
    if qual_w_t is not None:
        print(f"Qual weights:  {qual_w_t.detach().cpu().numpy().round(3).tolist()}")
    best_score = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()

        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        tr = run_epoch(
            model, train_loader, optimizer, scaler, cfg,
            num_types, num_qualities, train=True,
            type_w=type_w_t, qual_w=qual_w_t, cm_pack=None
        )

        if device_type == "cuda" and cfg.empty_cache_each_epoch:
            torch.cuda.empty_cache()

        cm_pack = None
        do_cm = (cfg.cm_every > 0) and (epoch % cfg.cm_every == 0)
        if not MPL_OK:
            do_cm = False

        if do_cm:
            cm_pack = {
                "cm_type_px": np.zeros((num_types, num_types), dtype=np.int64),
                "cm_qual_px": np.zeros((num_qualities, num_qualities), dtype=np.int64),
                "cm_type_obj": np.zeros((num_types, num_types), dtype=np.int64),
                "cm_qual_obj": np.zeros((num_qualities, num_qualities), dtype=np.int64),
            }

        va = run_epoch(
            model, val_loader, None, None, cfg,
            num_types, num_qualities, train=False,
            type_w=type_w_t, qual_w=qual_w_t, cm_pack=cm_pack
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        dt = time.perf_counter() - t0


        score = 0.35 * va["fruit_iou"] + 0.35 * va["quality_acc"] + 0.30 * va["type_acc"]

        peak_mb = 0.0
        alloc_mb, reserv_mb = 0.0, 0.0
        if device_type == "cuda":
            peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            alloc_mb, reserv_mb = cuda_mem_mb()

        header = (
            f"Epoch {epoch:03d}/{cfg.epochs} | lr={lr:.2e} | time={dt:.1f}s | score={score:.4f}"
            + (f" | peak_alloc={peak_mb:.0f}MB alloc={alloc_mb:.0f}MB reserv={reserv_mb:.0f}MB" if device_type == "cuda" else "")
        )
        body = (
            f"  train: loss={tr['loss']:.4f} fruit_iou={tr['fruit_iou']:.4f} type_acc={tr['type_acc']:.4f} qual_acc={tr['quality_acc']:.4f}\n"
            f"  val:   loss={va['loss']:.4f} fruit_iou={va['fruit_iou']:.4f} fruit_dice={va['fruit_dice']:.4f} "
            f"type_acc={va['type_acc']:.4f} qual_acc={va['quality_acc']:.4f} "
            f"type_mIoU={va['type_miou']:.4f} qual_mIoU={va['quality_miou']:.4f}"
        )
        print(header)
        print(body)

        history.append({"epoch": epoch, "lr": lr, "train": tr, "val": va, "score": score})


        if do_cm and cm_pack is not None:
            out_cms = os.path.join(cfg.out_dir, "cms")
            save_cm_png(cm_pack["cm_qual_obj"], train_ds.idx.quality_names, f"Quality CM (object) [val] e{epoch:03d}", os.path.join(out_cms, f"e{epoch:03d}_qual_obj.png"))
            save_cm_png(cm_pack["cm_qual_px"], train_ds.idx.quality_names, f"Quality CM (pixel) [val] e{epoch:03d}", os.path.join(out_cms, f"e{epoch:03d}_qual_px.png"))
            save_cm_png(cm_pack["cm_type_obj"], train_ds.idx.type_names, f"Type CM (object) [val] e{epoch:03d}", os.path.join(out_cms, f"e{epoch:03d}_type_obj.png"))
            save_cm_png(cm_pack["cm_type_px"], train_ds.idx.type_names, f"Type CM (pixel) [val] e{epoch:03d}", os.path.join(out_cms, f"e{epoch:03d}_type_px.png"))


        if score > best_score:
            best_score = score
            ckpt_path = os.path.join(cfg.out_dir, "weights", "best.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "type_names": train_ds.idx.type_names,
                    "quality_names": train_ds.idx.quality_names,
                },
                ckpt_path,
            )
            print(f"  [BEST] saved: {ckpt_path}")


        if epoch % 10 == 0 or epoch == cfg.epochs:
            ckpt_path = os.path.join(cfg.out_dir, "weights", f"epoch_{epoch:03d}.pth")
            torch.save({"model": model.state_dict()}, ckpt_path)


        if cfg.vis_every > 0 and (epoch % cfg.vis_every == 0 or epoch == 1):
            model.eval()
            with torch.inference_mode():
                batch = next(iter(val_loader))
                imgs, _fruit_gt, _type_gt, _qual_gt, names = batch
                imgs = imgs.to(cfg.device, non_blocking=True)
                if device_type == "cuda" and cfg.use_channels_last:
                    imgs = imgs.to(memory_format=torch.channels_last)

                if use_amp:
                    with autocast(device_type=device_type, enabled=True):
                        fruit_logits, type_logits, qual_logits = model(imgs)
                else:
                    fruit_logits, type_logits, qual_logits = model(imgs)

                save_vis_batch(
                    out_dir=os.path.join(cfg.out_dir, "vis"),
                    epoch=epoch,
                    names=list(names),
                    imgs=imgs,
                    fruit_logits=fruit_logits,
                    type_logits=type_logits,
                    qual_logits=qual_logits,
                    max_vis=cfg.max_vis,
                )

                del imgs, fruit_logits, type_logits, qual_logits

        if device_type == "cuda" and cfg.empty_cache_each_epoch:
            torch.cuda.empty_cache()

    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Готово. Лучший результат = {best_score:.4f}")
    print(f"Выходы: {cfg.out_dir}")


if __name__ == "__main__":
    main()