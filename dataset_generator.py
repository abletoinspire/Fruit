import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def poly_to_mask(polys: List[List[float]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def mask_to_polygons(mask: np.ndarray, simplify_eps: float = 2.0) -> List[List[float]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[List[float]] = []
    for c in cnts:
        if len(c) < 6:
            continue
        approx = cv2.approxPolyDP(c, simplify_eps, True)
        if len(approx) < 6:
            continue
        poly = approx.reshape(-1, 2).astype(np.float32)
        out.append(poly.flatten().tolist())
    return out


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def rotate_rgba(rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    cX, cY = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rotated = cv2.warpAffine(
        rgba, M, (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


def resize_rgba(rgba: np.ndarray, scale: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    nh = max(2, int(round(h * scale)))
    nw = max(2, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(rgba, (nw, nh), interpolation=interp)


def flip_rgba(rgba: np.ndarray, horizontal: bool = False, vertical: bool = False) -> np.ndarray:
    out = rgba
    if horizontal and vertical:
        out = cv2.flip(out, -1)
    elif horizontal:
        out = cv2.flip(out, 1)
    elif vertical:
        out = cv2.flip(out, 0)
    return out


def alpha_blend(bg_bgr: np.ndarray, fg_rgba: np.ndarray, x: int, y: int, add_shadow: bool = True):
    h, w = fg_rgba.shape[:2]
    H, W = bg_bgr.shape[:2]

    x1, y1 = int(x), int(y)
    x2, y2 = x1 + w, y1 + h

    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return

    bx1 = max(0, min(W, x1))
    by1 = max(0, min(H, y1))
    bx2 = max(0, min(W, x2))
    by2 = max(0, min(H, y2))

    fx1 = bx1 - x1
    fy1 = by1 - y1
    fx2 = fx1 + (bx2 - bx1)
    fy2 = fy1 + (by2 - by1)

    roi = bg_bgr[by1:by2, bx1:bx2].astype(np.float32)
    fg = fg_rgba[fy1:fy2, fx1:fx2].astype(np.float32)

    alpha = fg[:, :, 3:4] / 255.0
    rgb = fg[:, :, :3]

    if add_shadow:
        sh = (fg[:, :, 3] > 0).astype(np.uint8) * 255
        sh = cv2.GaussianBlur(sh, (0, 0), sigmaX=6, sigmaY=6)
        sh = sh.astype(np.float32) / 255.0
        sh = sh[:, :, None]
        shadow_strength = 0.18
        roi = roi * (1.0 - shadow_strength * sh)

    out = roi * (1.0 - alpha) + rgb * alpha
    bg_bgr[by1:by2, bx1:bx2] = out.astype(np.uint8)


def fruit_type_from_catname(name: str) -> str:
    n = (name or "").lower().strip()

    if "banana" in n:
        return "banana"
    if "apple" in n:
        return "apple"
    if "mandarin" in n:
        return "mandarin"
    if "tangerine" in n:
        return "mandarin"
    if "orange" in n:
        return "orange"

    parts = n.split("_")
    if len(parts) >= 2:
        return parts[-1]
    return n if n else "unknown"


@dataclass
class CropItem:
    rgba: np.ndarray
    category_id: int
    fruit_type: str


def extract_crops_from_coco(
    coco: dict,
    images_dir: str,
    bbox_expand: float = 1.15,
    min_side: int = 32,
) -> List[CropItem]:
    images = {im["id"]: im for im in coco.get("images", [])}
    anns = coco.get("annotations", [])
    catid_to_name = {c["id"]: c.get("name", "") for c in coco.get("categories", [])}

    crops: List[CropItem] = []

    for ann in anns:
        if ann.get("iscrowd", 0) == 1:
            continue

        img_info = images.get(ann.get("image_id"))
        if img_info is None:
            continue

        img_path = os.path.join(images_dir, img_info["file_name"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        H, W = img.shape[:2]
        seg = ann.get("segmentation", [])
        if not seg or not isinstance(seg, list):
            continue

        mask = poly_to_mask(seg, H, W)
        if mask.sum() == 0:
            continue

        x, y, w, h = ann["bbox"]
        cx, cy = x + w / 2.0, y + h / 2.0
        ww, hh = w * bbox_expand, h * bbox_expand

        x1 = int(max(0, cx - ww / 2.0))
        y1 = int(max(0, cy - hh / 2.0))
        x2 = int(min(W, cx + ww / 2.0))
        y2 = int(min(H, cy + hh / 2.0))

        if x2 - x1 < min_side or y2 - y1 < min_side:
            continue

        crop_rgb = img[y1:y2, x1:x2].copy()
        crop_m = mask[y1:y2, x1:x2].copy()

        rgba = np.dstack([crop_rgb, crop_m]).astype(np.uint8)


        tx1, ty1, tx2, ty2 = bbox_from_mask(crop_m)
        if tx2 > tx1 and ty2 > ty1:
            rgba = rgba[ty1:ty2 + 1, tx1:tx2 + 1]

        cid = int(ann["category_id"])
        cname = catid_to_name.get(cid, "")
        ftype = fruit_type_from_catname(cname)

        crops.append(CropItem(rgba=rgba, category_id=cid, fruit_type=ftype))

    return crops


def choose_target_max_side_px(fruit_type: str, belt_h: int) -> float:
    ft = (fruit_type or "").lower()

    if ft == "apple":
        return random.uniform(0.35, 0.55) * belt_h
    if ft in ("mandarin", "orange"):
        return random.uniform(0.30, 0.50) * belt_h
    if ft == "banana":
        return random.uniform(0.55, 0.85) * belt_h

    return random.uniform(0.35, 0.55) * belt_h


def compute_scale_by_belt_height(item: CropItem, belt_h: int, y_rel: float) -> float:
    alpha = item.rgba[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        return 1.0

    obj_w = float(xs.max() - xs.min() + 1)
    obj_h = float(ys.max() - ys.min() + 1)
    obj_max = max(obj_w, obj_h)

    target = choose_target_max_side_px(item.fruit_type, belt_h)

    y_rel = clamp(float(y_rel), 0.0, 1.0)
    persp = 0.90 + 0.20 * y_rel
    target *= persp

    scale = target / max(1.0, obj_max)

    return clamp(scale, 0.08, 0.90)


def photometric_tweak_rgba(rgba: np.ndarray) -> np.ndarray:
    out = rgba.copy()
    rgb = out[:, :, :3].astype(np.float32)

    mul = random.uniform(0.92, 1.06)
    add = random.uniform(-6, 6)

    rgb = np.clip(rgb * mul + add, 0, 255)
    out[:, :, :3] = rgb.astype(np.uint8)
    return out


def pick_k(min_k: int, max_k: int, p_empty: float) -> int:
    if random.random() < p_empty:
        return 0
    return random.randint(min_k, max_k)


def place_on_belt(
    bg_bgr: np.ndarray,
    roi: Tuple[int, int, int, int],
    crop_items: List[CropItem],
    min_k: int,
    max_k: int,
    p_empty: float,
    allow_overlap: bool = False,
    hflip_p: float = 0.5,
    vflip_p: float = 0.0,
) -> Tuple[np.ndarray, List[dict]]:
    x_min, y_top, x_max, y_bot = map(int, roi)
    belt_w = x_max - x_min
    belt_h = y_bot - y_top

    out = bg_bgr.copy()
    annotations: List[dict] = []
    placed_masks = []

    k = pick_k(min_k, max_k, p_empty=p_empty)

    for _ in range(k):
        item = random.choice(crop_items)

        tries = 60
        ok = False

        for _t in range(tries):
            py_tmp = random.randint(y_top, max(y_top, y_bot - 2))
            y_rel = (py_tmp - y_top) / max(1.0, float(belt_h))

            scale = compute_scale_by_belt_height(item, belt_h, y_rel)


            do_h = (random.random() < float(hflip_p))
            do_v = (random.random() < float(vflip_p))

            rgba_src = item.rgba
            if do_h or do_v:
                rgba_src = flip_rgba(rgba_src, horizontal=do_h, vertical=do_v)

            rgba2 = resize_rgba(rgba_src, scale)


            if item.fruit_type == "banana":
                angle = random.uniform(-18, 18)
            else:
                angle = random.uniform(-12, 12)

            rgba2 = rotate_rgba(rgba2, angle)

            h, w = rgba2.shape[:2]
            if w >= belt_w or h >= belt_h:
                continue

            px = random.randint(x_min, x_max - w)
            py = random.randint(y_top, y_bot - h)

            alpha = rgba2[:, :, 3]
            mask_local = (alpha > 10).astype(np.uint8)

            if not allow_overlap and placed_masks:
                rx1, ry1 = px - x_min, py - y_top
                rx2, ry2 = rx1 + w, ry1 + h

                overlap_found = False
                for (ax1, ay1, ax2, ay2, amask) in placed_masks:
                    ix1 = max(rx1, ax1)
                    iy1 = max(ry1, ay1)
                    ix2 = min(rx2, ax2)
                    iy2 = min(ry2, ay2)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    a_cut = amask[iy1 - ay1:iy2 - ay1, ix1 - ax1:ix2 - ax1]
                    b_cut = mask_local[iy1 - ry1:iy2 - ry1, ix1 - rx1:ix2 - rx1]
                    if (a_cut & b_cut).sum() > 0:
                        overlap_found = True
                        break

                if overlap_found:
                    continue

            ok = True
            break

        if not ok:
            continue

        rgba2 = photometric_tweak_rgba(rgba2)
        alpha_blend(out, rgba2, px, py, add_shadow=True)

        mask_bin = (rgba2[:, :, 3] > 10).astype(np.uint8) * 255
        polys = mask_to_polygons(mask_bin, simplify_eps=2.0)
        if not polys:
            continue

        shifted_polys: List[List[float]] = []
        for p in polys:
            pp = np.array(p, dtype=np.float32).reshape(-1, 2)
            pp[:, 0] += px
            pp[:, 1] += py
            shifted_polys.append(pp.flatten().tolist())

        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        bx1, bx2 = int(xs.min() + px), int(xs.max() + px)
        by1, by2 = int(ys.min() + py), int(ys.max() + py)
        bw = bx2 - bx1 + 1
        bh = by2 - by1 + 1
        area = int((mask_bin > 0).sum())

        annotations.append({
            "category_id": item.category_id,
            "segmentation": shifted_polys,
            "bbox": [bx1, by1, bw, bh],
            "area": area,
            "iscrowd": 0,
        })

        if not allow_overlap:
            rx1, ry1 = px - x_min, py - y_top
            placed_masks.append((rx1, ry1, rx1 + w, ry1 + h, (mask_bin > 10).astype(np.uint8)))

    return out, annotations


def build_dataset(
    split_name: str,
    coco_path: str,
    coco_images_dir: str,
    backgrounds_dir: str,
    rois_cfg: dict,
    out_root: str,
    n_images: int,
    seed: int,
    min_k: int,
    max_k: int,
    p_empty: float,
    bbox_expand: float,
    min_crop_side: int,
    hflip_p: float,
    vflip_p: float,
):
    set_seed(seed)

    coco = load_json(coco_path)
    categories = coco.get("categories", [])

    crop_pool = extract_crops_from_coco(
        coco=coco,
        images_dir=coco_images_dir,
        bbox_expand=bbox_expand,
        min_side=min_crop_side,
    )

    if not crop_pool:
        raise RuntimeError(
            f"[{split_name}] Не удалось извлечь кропы. Проверь пути к COCO/изображениям и наличие segmentation."
        )

    bg_map = rois_cfg.get("backgrounds", {})
    bg_names = sorted(bg_map.keys())
    if not bg_names:
        raise ValueError("В ROI-конфиге нет записей backgrounds.")

    out_img_dir = os.path.join(out_root, split_name, "images")
    ensure_dir(out_img_dir)

    out_coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    ann_id = 1
    for i in range(n_images):
        bg_name = random.choice(bg_names)
        bg_path = os.path.join(backgrounds_dir, bg_name)
        bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        if bg is None:
            raise FileNotFoundError(bg_path)

        roi = tuple(bg_map[bg_name]["roi"])

        composed, anns = place_on_belt(
            bg_bgr=bg,
            roi=roi,
            crop_items=crop_pool,
            min_k=min_k,
            max_k=max_k,
            p_empty=p_empty,
            allow_overlap=False,
            hflip_p=hflip_p,
            vflip_p=vflip_p,
        )

        file_name = f"{split_name}_{i:06d}.png"
        out_path = os.path.join(out_img_dir, file_name)
        cv2.imwrite(out_path, composed)

        img_h, img_w = composed.shape[:2]
        img_id = i + 1

        out_coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": img_w,
            "height": img_h,
            "bg_name": bg_name,
        })

        for a in anns:
            a["id"] = ann_id
            a["image_id"] = img_id
            out_coco["annotations"].append(a)
            ann_id += 1

    out_ann_path = os.path.join(out_root, split_name, "annotations.json")
    ensure_dir(os.path.dirname(out_ann_path))
    with open(out_ann_path, "w", encoding="utf-8") as f:
        json.dump(out_coco, f, ensure_ascii=False)

    print(f"[{split_name}] Готово.")
    print(f"  Изображений создано: {n_images}")
    
    print(f"  Аннотаций создано: {len(out_coco['annotations'])}")
    
    print(f"  Сохранено: {out_ann_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois-json", required=True, help="JSON-файл с ROI для фонов")
    ap.add_argument("--backgrounds-dir", required=True, help="Папка с фоновыми изображениями")

    ap.add_argument("--train-coco", required=True, help="COCO annotations.json для train")
    ap.add_argument("--train-images", required=True, help="Папка train-изображений")
    ap.add_argument("--val-coco", required=True, help="COCO annotations.json для val")
    ap.add_argument("--val-images", required=True, help="Папка val-изображений")

    ap.add_argument("--out", required=True, help="Корневая папка для результата")
    ap.add_argument("--n-train", type=int, default=320)
    ap.add_argument("--n-val", type=int, default=119)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--p-empty", type=float, default=0.08, help="Вероятность изображения без фруктов")

    ap.add_argument("--bbox-expand", type=float, default=1.15, help="Коэффициент расширения bbox при извлечении кропа из COCO")
    ap.add_argument("--min-crop-side", type=int, default=32, help="Пропускать слишком маленькие кропы")


    ap.add_argument("--hflip-p", type=float, default=0.50, help="Вероятность горизонтального отражения для каждого вставленного кропа")
    ap.add_argument("--vflip-p", type=float, default=0.00, help="Вероятность вертикального отражения для каждого вставленного кропа")

    args = ap.parse_args()

    rois = load_json(args.rois_json)

    build_dataset(
        split_name="train",
        coco_path=args.train_coco,
        coco_images_dir=args.train_images,
        backgrounds_dir=args.backgrounds_dir,
        rois_cfg=rois,
        out_root=args.out,
        n_images=args.n_train,
        seed=args.seed,
        min_k=args.min_k,
        max_k=args.max_k,
        p_empty=args.p_empty,
        bbox_expand=args.bbox_expand,
        min_crop_side=args.min_crop_side,
        hflip_p=float(args.hflip_p),
        vflip_p=float(args.vflip_p),
    )

    build_dataset(
        split_name="val",
        coco_path=args.val_coco,
        coco_images_dir=args.val_images,
        backgrounds_dir=args.backgrounds_dir,
        rois_cfg=rois,
        out_root=args.out,
        n_images=args.n_val,
        seed=args.seed + 1,
        min_k=args.min_k,
        max_k=args.max_k,
        p_empty=args.p_empty,
        bbox_expand=args.bbox_expand,
        min_crop_side=args.min_crop_side,
        hflip_p=float(args.hflip_p),
        vflip_p=float(args.vflip_p),
    )


if __name__ == "__main__":
    main()