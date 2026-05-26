import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

@dataclass
class UserPaths:
    coco_path: str = r"D:\YOLOFRUIT\dataset_generator\annotations\instances_Train.json"
    fruits_dir: str = r"D:\YOLOFRUIT\dataset_generator\images\Train"
    bgs_dir: str = r"D:\YOLOFRUIT\dataset_generator\bgs"
    out_dir: str = r"D:\YOLOFRUIT\dataset_generator\synthetic"

@dataclass
class UserParams:
    num: int = 5000
    out_w: int = 640
    out_h: int = 640
    seed: int = 42

    merge_by_suffix: bool = True

    allow_hflip_bg: bool = True
    allow_vflip_bg: bool = False
    bg_blur_prob: float = 0.35
    bg_noise_prob: float = 0.35

    rotate_deg_max: float = 40.0
    allow_hflip_fruit: bool = True
    allow_vflip_fruit: bool = False
    scale_min: float = 0.50
    scale_max: float = 1.30

    k_min: int = 2
    k_max: int = 4

USER_PATHS = UserPaths()
USER_PARAMS = UserParams()

@dataclass
class GenConfig:
    coco_path: str
    fruits_dir: str
    bgs_dir: str
    out_dir: str

    num: int
    out_w: int
    out_h: int
    seed: int

    k_min: int = 2
    k_max: int = 4
    sector_margin_frac: float = 0.08

    fruits_per_sector_min: int = 1
    fruits_per_sector_max: int = 1
    max_tries_place: int = 25
    max_overlap_ratio: float = 0.10

    rotate_deg_max: float = 40.0
    allow_hflip_fruit: bool = True
    allow_vflip_fruit: bool = False
    scale_min: float = 0.50
    scale_max: float = 1.30
    target_min_frac: float = 0.45
    target_max_frac: float = 0.85

    allow_hflip_bg: bool = True
    allow_vflip_bg: bool = False
    bg_blur_prob: float = 0.35
    bg_blur_ksize_min: int = 3
    bg_blur_ksize_max: int = 11
    bg_noise_prob: float = 0.35
    bg_noise_sigma_min: float = 3.0
    bg_noise_sigma_max: float = 18.0

    feather_edges: bool = True
    feather_px_min: int = 1
    feather_px_max: int = 3

    min_sector_side: int = 120

    merge_by_suffix: bool = True
    allowed_base_classes: Optional[List[str]] = None

def make_cfg() -> GenConfig:
    cfg = GenConfig(
        coco_path=USER_PATHS.coco_path,
        fruits_dir=USER_PATHS.fruits_dir,
        bgs_dir=USER_PATHS.bgs_dir,
        out_dir=USER_PATHS.out_dir,
        num=USER_PARAMS.num,
        out_w=USER_PARAMS.out_w,
        out_h=USER_PARAMS.out_h,
        seed=USER_PARAMS.seed,
    )

    cfg.merge_by_suffix = USER_PARAMS.merge_by_suffix

    cfg.allow_hflip_bg = USER_PARAMS.allow_hflip_bg
    cfg.allow_vflip_bg = USER_PARAMS.allow_vflip_bg
    cfg.bg_blur_prob = USER_PARAMS.bg_blur_prob
    cfg.bg_noise_prob = USER_PARAMS.bg_noise_prob

    cfg.rotate_deg_max = USER_PARAMS.rotate_deg_max
    cfg.allow_hflip_fruit = USER_PARAMS.allow_hflip_fruit
    cfg.allow_vflip_fruit = USER_PARAMS.allow_vflip_fruit
    cfg.scale_min = USER_PARAMS.scale_min
    cfg.scale_max = USER_PARAMS.scale_max

    cfg.k_min = USER_PARAMS.k_min
    cfg.k_max = USER_PARAMS.k_max

    return cfg

@dataclass
class Cutout:
    cls_id: int
    cls_name: str
    rgb: np.ndarray
    mask: np.ndarray

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                out.append(os.path.join(root, f))
    out.sort()
    return out

def load_coco(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def base_class_name(name: str, merge_by_suffix: bool) -> str:
    n = name.strip().lower()
    if merge_by_suffix and "_" in n:
        return n.split("_")[-1]
    return n

def polygon_to_mask(h: int, w: int, seg) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    if seg is None:
        return mask

    polys = []
    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
        polys = seg
    elif isinstance(seg, list):
        polys = [seg]
    else:
        return mask

    for poly in polys:
        if not poly or len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts_i = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts_i], 255)

    return mask

def mask_to_polygons(mask: np.ndarray, min_area: int = 20) -> List[List[float]]:
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, 0.002 * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if approx.shape[0] < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float32)
        poly = pts.reshape(-1).tolist()
        if len(poly) >= 6:
            polys.append(poly)
    return polys

def crop_cutout(img_bgr: np.ndarray, mask_full: np.ndarray, pad: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_full > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(mask_full.shape[1] - 1, x2 + pad)
    y2 = min(mask_full.shape[0] - 1, y2 + pad)

    crop_bgr = img_bgr[y1:y2 + 1, x1:x2 + 1]
    crop_mask = mask_full[y1:y2 + 1, x1:x2 + 1]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return crop_rgb, crop_mask

def build_cutouts(coco: dict, fruits_dir: str, cfg: GenConfig) -> Tuple[List[Cutout], Dict[str, int], Dict[int, str]]:
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    cats = coco.get("categories", [])
    catid_to_name = {c["id"]: c["name"] for c in cats}

    base_names = set()
    for nm in catid_to_name.values():
        base_names.add(base_class_name(nm, cfg.merge_by_suffix))
    base_names = sorted(base_names)

    if cfg.allowed_base_classes is None:
        preferred = ["apple", "banana", "orange", "mandarin"]
        present = [p for p in preferred if p in base_names]
        if present:
            base_names = present

    name_to_newid = {nm: i + 1 for i, nm in enumerate(base_names)}
    newid_to_name = {i + 1: nm for i, nm in enumerate(base_names)}

    cutouts: List[Cutout] = []
    img_cache: Dict[int, np.ndarray] = {}

    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        cat_name = catid_to_name.get(cat_id, None)
        if not cat_name:
            continue

        base = base_class_name(cat_name, cfg.merge_by_suffix)
        if base not in name_to_newid:
            continue

        im = images_by_id.get(img_id, None)
        if im is None:
            continue

        img_path = os.path.join(fruits_dir, im.get("file_name", ""))
        if not os.path.exists(img_path):
            continue

        if img_id not in img_cache:
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            img_cache[img_id] = bgr
        else:
            bgr = img_cache[img_id]

        h, w = bgr.shape[:2]
        seg = ann.get("segmentation", None)
        if not isinstance(seg, list):
            continue                       

        mask_full = polygon_to_mask(h, w, seg)
        if mask_full.sum() == 0:
            continue

        crop_rgb, crop_mask = crop_cutout(bgr, mask_full, pad=2)
        if crop_rgb is None:
            continue

        cls_id = name_to_newid[base]
        cutouts.append(Cutout(cls_id=cls_id, cls_name=base, rgb=crop_rgb, mask=crop_mask))

    if len(cutouts) == 0:
        raise RuntimeError("Не удалось извлечь ни одного cutout. Нужны polygon segmentation в COCO.")

    return cutouts, name_to_newid, newid_to_name

def resize_crop_to(bg_bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = bg_bgr.shape[:2]
    target_ar = out_w / out_h
    ar = w / h

    if ar > target_ar:
        new_w = int(h * target_ar)
        x1 = (w - new_w) // 2
        crop = bg_bgr[:, x1:x1 + new_w]
    else:
        new_h = int(w / target_ar)
        y1 = (h - new_h) // 2
        crop = bg_bgr[y1:y1 + new_h, :]

    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

def apply_bg_transforms(bg_bgr: np.ndarray, cfg: GenConfig, rng: random.Random) -> np.ndarray:
    out = bg_bgr.copy()

    if cfg.allow_hflip_bg and rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if cfg.allow_vflip_bg and rng.random() < 0.2:
        out = cv2.flip(out, 0)

    if rng.random() < cfg.bg_blur_prob:
        k = rng.randrange(cfg.bg_blur_ksize_min, cfg.bg_blur_ksize_max + 1, 2)
        out = cv2.GaussianBlur(out, (k, k), 0)

    if rng.random() < cfg.bg_noise_prob:
        sigma = rng.uniform(cfg.bg_noise_sigma_min, cfg.bg_noise_sigma_max)
        n = np.random.normal(0, sigma, size=out.shape).astype(np.float32)
        out_f = out.astype(np.float32) + n
        out = np.clip(out_f, 0, 255).astype(np.uint8)

    return out

Rect = Tuple[int, int, int, int]           

def rect_area(r: Rect) -> int:
    return r[2] * r[3]

def guillotine_split(out_w: int, out_h: int, k: int, cfg: GenConfig, rng: random.Random) -> List[Rect]:
    rects: List[Rect] = [(0, 0, out_w, out_h)]
    min_side = cfg.min_sector_side

    while len(rects) < k:
        idx = max(range(len(rects)), key=lambda i: rect_area(rects[i]))
        x, y, w, h = rects.pop(idx)

        can_v = w >= 2 * min_side
        can_h = h >= 2 * min_side
        if not can_v and not can_h:
            rects.append((x, y, w, h))
            break

        if can_v and not can_h:
            dir_v = True
        elif can_h and not can_v:
            dir_v = False
        else:
            if w / h > 1.25:
                dir_v = True
            elif h / w > 1.25:
                dir_v = False
            else:
                dir_v = rng.random() < 0.5

        if dir_v:
            cut = rng.randint(min_side, w - min_side)
            r1 = (x, y, cut, h)
            r2 = (x + cut, y, w - cut, h)
        else:
            cut = rng.randint(min_side, h - min_side)
            r1 = (x, y, w, cut)
            r2 = (x, y + cut, w, h - cut)

        rects.extend([r1, r2])

    if len(rects) < k:
        rects = []
        cols = 2
        rows = (k + cols - 1) // cols
        cell_w = out_w // cols
        cell_h = out_h // rows
        for rr in range(rows):
            for cc in range(cols):
                if len(rects) >= k:
                    break
                x = cc * cell_w
                y = rr * cell_h
                w = cell_w if cc < cols - 1 else out_w - x
                h = cell_h if rr < rows - 1 else out_h - y
                rects.append((x, y, w, h))

    return rects

def random_point_in_rect(r: Rect, margin_frac: float, rng: random.Random) -> Tuple[int, int]:
    x, y, w, h = r
    m = int(min(w, h) * margin_frac)
    if w - 2 * m <= 1 or h - 2 * m <= 1:
        m = 0
    px = rng.randint(x + m, x + w - 1 - m)
    py = rng.randint(y + m, y + h - 1 - m)
    return px, py

def feather_alpha(mask: np.ndarray, px: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if px <= 0:
        return m.astype(np.float32)

    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    alpha = dist_in / (dist_in + dist_out + 1e-6)
    alpha = np.clip(alpha * (1.0 + px * 0.15), 0.0, 1.0)
    return alpha.astype(np.float32)

def transform_fruit(rgb: np.ndarray, mask: np.ndarray, cfg: GenConfig, rng: random.Random,
                    target_side: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = rgb.copy()
    m = mask.copy()

    if cfg.allow_hflip_fruit and rng.random() < 0.5:
        img = cv2.flip(img, 1)
        m = cv2.flip(m, 1)
    if cfg.allow_vflip_fruit and rng.random() < 0.15:
        img = cv2.flip(img, 0)
        m = cv2.flip(m, 0)

    h, w = m.shape[:2]
    cur_side = max(h, w)
    if cur_side < 1:
        return img, m, (m > 0).astype(np.float32)

    scale_to = target_side / cur_side
    scale_jit = rng.uniform(cfg.scale_min, cfg.scale_max)
    scale = scale_to * scale_jit

    new_w = max(2, int(w * scale))
    new_h = max(2, int(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    angle = rng.uniform(-cfg.rotate_deg_max, cfg.rotate_deg_max)
    hh, ww = m.shape[:2]
    cx, cy = ww / 2.0, hh / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    bound_w = int(hh * sin + ww * cos)
    bound_h = int(hh * cos + ww * sin)

    M[0, 2] += bound_w / 2 - cx
    M[1, 2] += bound_h / 2 - cy

    img_r = cv2.warpAffine(img, M, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    m_r = cv2.warpAffine(m, M, (bound_w, bound_h), flags=cv2.INTER_NEAREST, borderValue=0)
    m_r = (m_r > 0).astype(np.uint8) * 255

    if cfg.feather_edges:
        px = rng.randint(cfg.feather_px_min, cfg.feather_px_max)
        alpha = feather_alpha(m_r, px)
    else:
        alpha = (m_r > 0).astype(np.float32)

    return img_r, m_r, alpha

def paste_rgba(bg_rgb: np.ndarray, fg_rgb: np.ndarray, fg_mask: np.ndarray, fg_alpha: np.ndarray,
               x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
    H, W = bg_rgb.shape[:2]
    h, w = fg_mask.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return bg_rgb, np.zeros((H, W), dtype=np.uint8)

    fx1 = x1 - x
    fy1 = y1 - y
    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)

    roi_bg = bg_rgb[y1:y2, x1:x2].astype(np.float32)
    roi_fg = fg_rgb[fy1:fy2, fx1:fx2].astype(np.float32)
    roi_a = fg_alpha[fy1:fy2, fx1:fx2].astype(np.float32)[..., None]

    out_roi = roi_bg * (1.0 - roi_a) + roi_fg * roi_a
    bg_rgb_new = bg_rgb.copy()
    bg_rgb_new[y1:y2, x1:x2] = np.clip(out_roi, 0, 255).astype(np.uint8)

    placed = np.zeros((H, W), dtype=np.uint8)
    placed_roi = fg_mask[fy1:fy2, fx1:fx2]
    placed[y1:y2, x1:x2] = np.maximum(placed[y1:y2, x1:x2], placed_roi)

    return bg_rgb_new, placed

def overlap_ok(existing: np.ndarray, new_mask: np.ndarray, max_ratio: float) -> bool:
    if new_mask.sum() == 0:
        return False
    inter = np.logical_and(existing > 0, new_mask > 0).sum()
    area = (new_mask > 0).sum()
    if area <= 0:
        return False
    return (inter / area) <= max_ratio

def validate_paths(cfg: GenConfig) -> None:
    assert os.path.exists(cfg.coco_path), f"COCO не найден: {cfg.coco_path}"
    assert os.path.isdir(cfg.fruits_dir), f"fruits_dir не папка: {cfg.fruits_dir}"
    assert os.path.isdir(cfg.bgs_dir), f"bgs_dir не папка: {cfg.bgs_dir}"
    ensure_dir(cfg.out_dir)

def generate(cfg: GenConfig) -> None:
    validate_paths(cfg)

    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    coco = load_coco(cfg.coco_path)
    cutouts, _, id_to_name = build_cutouts(coco, cfg.fruits_dir, cfg)

    by_cls: Dict[int, List[Cutout]] = {}
    for c in cutouts:
        by_cls.setdefault(c.cls_id, []).append(c)

    class_ids = sorted(by_cls.keys())
    gen_counts = {cid: 0 for cid in class_ids}

    bg_paths = list_images(cfg.bgs_dir)
    if len(bg_paths) == 0:
        raise RuntimeError("Не найдено фонов в bgs_dir.")

    out_images_dir = os.path.join(cfg.out_dir, "images")
    ensure_dir(out_images_dir)

    coco_out = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": id_to_name[cid]} for cid in sorted(id_to_name.keys())],
    }

    ann_id = 1
    img_id = 1

    for i in range(cfg.num):
        bg_path = bg_paths[i % len(bg_paths)]
        bg_bgr = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        if bg_bgr is None:
            continue

        bg_bgr = resize_crop_to(bg_bgr, cfg.out_w, cfg.out_h)
        bg_bgr = apply_bg_transforms(bg_bgr, cfg, rng)
        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)

        k = rng.randint(cfg.k_min, cfg.k_max)
        sectors = guillotine_split(cfg.out_w, cfg.out_h, k, cfg, rng)

        existing_mask = np.zeros((cfg.out_h, cfg.out_w), dtype=np.uint8)
        anns_for_img: List[dict] = []

        for sec in sectors:
            weights = [1.0 / (1.0 + gen_counts[cid]) for cid in class_ids]
            cls_id = rng.choices(class_ids, weights=weights, k=1)[0]

            c = by_cls[cls_id][rng.randrange(len(by_cls[cls_id]))]

            _, _, sw, sh = sec
            target_side = int(min(sw, sh) * rng.uniform(cfg.target_min_frac, cfg.target_max_frac))
            target_side = max(10, target_side)

            fg_rgb, fg_mask, fg_alpha = transform_fruit(c.rgb, c.mask, cfg, rng, target_side)
            fh, fw = fg_mask.shape[:2]
            if fh < 3 or fw < 3:
                continue

            ax, ay = random_point_in_rect(sec, cfg.sector_margin_frac, rng)

            placed_ok = False
            best = None

            for _try in range(cfg.max_tries_place):
                jx = rng.randint(-int(sw * 0.10), int(sw * 0.10))
                jy = rng.randint(-int(sh * 0.10), int(sh * 0.10))
                px = ax + jx
                py = ay + jy

                x = int(px - fw / 2)
                y = int(py - fh / 2)

                sx, sy, sw2, sh2 = sec
                x = max(sx, min(x, sx + sw2 - fw))
                y = max(sy, min(y, sy + sh2 - fh))

                tmp_bg, placed_mask = paste_rgba(bg_rgb, fg_rgb, fg_mask, fg_alpha, x, y)
                if placed_mask.sum() == 0:
                    continue
                if not overlap_ok(existing_mask, placed_mask, cfg.max_overlap_ratio):
                    continue

                best = (tmp_bg, placed_mask)
                placed_ok = True
                break

            if not placed_ok or best is None:
                continue

            bg_rgb, placed_mask = best
            existing_mask = np.maximum(existing_mask, placed_mask)

            polys = mask_to_polygons(placed_mask, min_area=30)
            if not polys:
                continue

            ys, xs = np.where(placed_mask > 0)
            if len(xs) == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bbox = [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]
            area = float((placed_mask > 0).sum())

            anns_for_img.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": bbox,
                "area": area,
                "segmentation": polys,
                "iscrowd": 0,
            })
            ann_id += 1
            gen_counts[cls_id] += 1

        file_name = f"syn_{i:06d}.jpg"
        out_path = os.path.join(out_images_dir, file_name)
        out_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        coco_out["images"].append({
            "id": img_id,
            "file_name": os.path.join("images", file_name).replace("\\", "/"),
            "width": cfg.out_w,
            "height": cfg.out_h,
        })
        coco_out["annotations"].extend(anns_for_img)

        img_id += 1

    out_ann = os.path.join(cfg.out_dir, "annotations.json")
    with open(out_ann, "w", encoding="utf-8") as f:
        json.dump(coco_out, f, ensure_ascii=False)

if __name__ == "__main__":
    cfg = make_cfg()
    generate(cfg)
