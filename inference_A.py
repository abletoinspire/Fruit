# python inference_A.py --weights .\runs_rotten_only\weights\best.pth --input .\test_images --output .\out_rotten_only --fruit-thr 0.55 --min-area 700 --split auto --core-erosion 5 --core-prob-thr 0.70 --core-min-px 200 --only-rotten --rotten-score-thr 0.55 --write-json --save-mask --save-prob --panel
import os
import cv2
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


AMP_AVAILABLE = False
try:
    from torch.amp import autocast
    AMP_AVAILABLE = True
except Exception:
    try:
        from torch.cuda.amp import autocast
        AMP_AVAILABLE = True
    except Exception:
        AMP_AVAILABLE = False


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


class UNetRottenOnly(nn.Module):
    def __init__(self, base_c: int, num_qualities: int = 2, head_drop: float = 0.10):
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
        self.head_rotten = SegClassHead(c1, num_qualities, drop_p=head_drop)

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

        return self.head_fruit(x), self.head_rotten(x)


def get_device_type(device: str) -> str:
    d = str(device).lower()
    if d.startswith("cuda"):
        return "cuda"
    if d == "mps":
        return "mps"
    return "cpu"


def letterbox(img_rgb: np.ndarray, out_size: int, pad_value: int = 0):
    h, w = img_rgb.shape[:2]
    scale = out_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_y = out_size - nh
    pad_x = out_size - nw
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left

    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value)
    )

    meta = {
        "orig_h": h, "orig_w": w,
        "scale": float(scale),
        "top": int(top), "left": int(left),
        "new_h": int(nh), "new_w": int(nw),
        "out_size": int(out_size),
    }
    return out, meta


def unletterbox_map(map_s: np.ndarray, meta: dict, H: int, W: int, is_mask: bool):
    top, left = meta["top"], meta["left"]
    nh, nw = meta["new_h"], meta["new_w"]
    out_size = meta["out_size"]

    crop = map_s[top:top + nh, left:left + nw]
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    full = cv2.resize(crop, (W, H), interpolation=interp)
    return full


def color_for_label(lbl: str):
    s = (lbl or "").lower()
    if "rotten" in s:
        return (0, 140, 255)
    if "fresh" in s:
        return (255, 180, 0)
    return (200, 200, 200)


def list_images(path: str):
    if os.path.isdir(path):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        files = []
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(exts):
                files.append(os.path.join(path, name))
        return files
    return [path]


def softmax_1d(x: np.ndarray):
    x = x.astype(np.float32)
    x = x - float(x.max())
    e = np.exp(x)
    s = float(e.sum()) + 1e-9
    return e / s


def mean_logits_on_mask(logits_chw: np.ndarray, mask_bool: np.ndarray):

    if logits_chw.ndim != 3:
        raise ValueError("logits_chw must be (C,H,W)")
    if not mask_bool.any():
        pooled = logits_chw.reshape(logits_chw.shape[0], -1).mean(axis=1)
    else:
        pooled = logits_chw[:, mask_bool].mean(axis=1)

    probs = softmax_1d(pooled)
    cls_id = int(np.argmax(probs))
    cls_conf = float(probs[cls_id])
    return cls_id, cls_conf, probs


def make_core_mask(comp_mask: np.ndarray,
                   fruit_prob_full: np.ndarray,
                   erosion: int,
                   prob_thr: float,
                   core_min_px: int) -> np.ndarray:
    if erosion <= 0:
        core = comp_mask.copy()
    else:
        u8 = comp_mask.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion))
        er = cv2.erode(u8, k, iterations=1)
        core = (er > 0)

    if fruit_prob_full is not None:
        core = core & (fruit_prob_full >= float(prob_thr))

    if int(core.sum()) < int(core_min_px):
        core2 = comp_mask.copy()
        if fruit_prob_full is not None:
            core2 = core2 & (fruit_prob_full >= float(prob_thr))
        if int(core2.sum()) >= int(core_min_px):
            return core2
        return comp_mask

    return core


def morph_open_only(mask_u8: np.ndarray, ksize: int = 5, iters: int = 1):
    if ksize <= 1:
        return mask_u8
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=iters)


def extract_cc_instances(bin_mask_u8: np.ndarray, min_area: int, do_open: bool = True):
    m = bin_mask_u8.copy()
    if do_open:
        m = morph_open_only(m, ksize=5, iters=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    comps = []
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        if area < min_area:
            continue
        comp_mask = (labels == lbl)
        comps.append((lbl, comp_mask, (int(x), int(y), int(w), int(h)), int(area)))
    return comps


def count_seeds_by_distance(mask_u8: np.ndarray, ws_fg_frac: float):
    opening = (mask_u8 > 0).astype(np.uint8)
    dist = cv2.distanceTransform(opening, distanceType=cv2.DIST_L2, maskSize=5)
    if dist.max() <= 1e-6:
        return 0, dist
    thr = float(ws_fg_frac) * float(dist.max())
    sure_fg = (dist >= thr).astype(np.uint8)
    n_markers, _ = cv2.connectedComponents(sure_fg)
    seeds = max(0, int(n_markers) - 1)
    return seeds, dist


def watershed_split_component(
    img_bgr: np.ndarray,
    comp_mask: np.ndarray,
    min_area: int,
    ws_fg_frac: float,
    ws_bg_dilate: int,
):
    H, W = img_bgr.shape[:2]
    ys, xs = np.where(comp_mask)
    if len(xs) == 0:
        return []

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    pad = 6
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(W - 1, x2 + pad)
    y2p = min(H - 1, y2 + pad)

    roi_img = img_bgr[y1p:y2p + 1, x1p:x2p + 1].copy()
    roi_mask = comp_mask[y1p:y2p + 1, x1p:x2p + 1].astype(np.uint8) * 255

    seeds, _ = count_seeds_by_distance(roi_mask, ws_fg_frac=ws_fg_frac)
    if seeds <= 1:
        comp_u8 = comp_mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []
        contour = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return [{
            "contour": contour,
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(comp_mask.sum()),
            "mask": comp_mask,
        }]

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate((roi_mask > 0).astype(np.uint8), k, iterations=int(ws_bg_dilate))

    opening = (roi_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(opening, distanceType=cv2.DIST_L2, maskSize=5)
    thr = float(ws_fg_frac) * float(dist.max())
    sure_fg = (dist >= thr).astype(np.uint8)

    unknown = cv2.subtract(sure_bg.astype(np.uint8), sure_fg.astype(np.uint8))

    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown > 0] = 0

    markers_ws = cv2.watershed(roi_img, markers.astype(np.int32))

    instances = []
    obj_ids = [mid for mid in np.unique(markers_ws) if mid >= 2]

    for mid in obj_ids:
        local = (markers_ws == mid)
        area = int(local.sum())
        if area < min_area:
            continue

        full_mask = np.zeros((H, W), dtype=bool)
        full_mask[y1p:y2p + 1, x1p:x2p + 1] = local

        comp_u8 = full_mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        contour = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        instances.append({
            "contour": contour,
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": area,
            "mask": full_mask,
        })

    if len(instances) == 0:
        comp_u8 = comp_mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []
        contour = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return [{
            "contour": contour,
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(comp_mask.sum()),
            "mask": comp_mask,
        }]

    return instances


def find_instances_auto(
    bin_mask_u8: np.ndarray,
    img_bgr: np.ndarray,
    min_area: int,
    use_open: bool,
    ws_fg_frac: float,
    ws_bg_dilate: int,
):
    comps = extract_cc_instances(bin_mask_u8, min_area=min_area, do_open=use_open)
    fruits = []
    for _, comp_mask, _, area in comps:

        ys, xs = np.where(comp_mask)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        roi = comp_mask[y1:y2 + 1, x1:x2 + 1].astype(np.uint8) * 255
        seeds, _ = count_seeds_by_distance(roi, ws_fg_frac=ws_fg_frac)
        if seeds <= 1:
            comp_u8 = comp_mask.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            contour = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(contour)
            fruits.append({
                "contour": contour,
                "bbox": (int(bx), int(by), int(bw), int(bh)),
                "area": int(area),
                "mask": comp_mask,
            })
        else:
            inst = watershed_split_component(
                img_bgr=img_bgr,
                comp_mask=comp_mask,
                min_area=min_area,
                ws_fg_frac=ws_fg_frac,
                ws_bg_dilate=ws_bg_dilate,
            )
            fruits.extend(inst)
    return fruits


def center_of_contour(contour):
    m = cv2.moments(contour)
    if abs(m.get("m00", 0.0)) < 1e-6:
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w * 0.5, y + h * 0.5)
    cx = float(m["m10"] / m["m00"])
    cy = float(m["m01"] / m["m00"])
    return (cx, cy)


def draw_side_panel(base_bgr: np.ndarray, dets: list):

    H, W = base_bgr.shape[:2]
    panel_w = max(520, int(W * 0.42))
    out = np.zeros((H, W + panel_w, 3), dtype=np.uint8)
    out[:, :W] = base_bgr
    out[:, W:] = (30, 30, 30)

    cv2.putText(out, f"Detections: {len(dets)}", (W + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    y0 = 80
    line_h = 36
    max_lines = max(1, (H - y0 - 20) // line_h)
    dets_show = dets[:max_lines]

    for i, d in enumerate(dets_show):
        y = y0 + i * line_h
        cv2.rectangle(out, (W + 20, y - 16), (W + 36, y), d["color"], -1)
        text = (
            f"#{d['idx']}  {d['quality']}  conf={d['conf']:.2f}  "
            f"(fruit={d['fruit_conf']:.2f}, pR={d['p_rotten']:.2f})"
        )
        cv2.putText(out, text, (W + 50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA)

        cx, cy = d["center_xy"]
        tx, ty = W + 20, y - 8
        cv2.line(out, (int(cx), int(cy)), (int(tx), int(ty)), (80, 160, 80), 1, cv2.LINE_AA)

    if len(dets) > len(dets_show):
        cv2.putText(out, f"... +{len(dets) - len(dets_show)} more", (W + 20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)

    return out


def infer_one(
    model: nn.Module,
    img_bgr: np.ndarray,
    img_size: int,
    quality_names,
    device: str,
    use_amp: bool,
    channels_last: bool,
    fruit_thr: float,
    min_area: int,
    split: str,
    ws_fg_frac: float,
    ws_bg_dilate: int,
    cls_thr: float,
    only_rotten: bool,
    rotten_score_thr: float,
    use_open: bool,
    core_erosion: int,
    core_prob_thr: float,
    core_min_px: int,
    panel: bool,
):
    H, W = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    lb, meta = letterbox(rgb, img_size, pad_value=0)
    x = torch.from_numpy(lb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device, non_blocking=True)

    device_type = get_device_type(device)
    if device_type == "cuda" and channels_last:
        x = x.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        if use_amp:
            with autocast(device_type=device_type, enabled=True):
                fruit_logits, qual_logits = model(x)
        else:
            fruit_logits, qual_logits = model(x)

        fruit_prob = torch.sigmoid(fruit_logits)[0, 0]
        qual_logits_s = qual_logits[0]

    fruit_prob_np = fruit_prob.float().cpu().numpy()
    qual_logits_np = qual_logits_s.float().cpu().numpy()

    fruit_prob_full = unletterbox_map(fruit_prob_np, meta, H, W, is_mask=False).astype(np.float32)
    qual_logits_full = np.stack(
        [unletterbox_map(qual_logits_np[q], meta, H, W, is_mask=False) for q in range(len(quality_names))],
        axis=0
    ).astype(np.float32)

    bin_mask = (fruit_prob_full >= float(fruit_thr)).astype(np.uint8) * 255

    split = split.lower()
    if split == "cc":
        comps = extract_cc_instances(bin_mask, min_area=min_area, do_open=use_open)
        fruits = []
        for _, comp_mask, _, area in comps:
            comp_u8 = comp_mask.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            contour = max(cnts, key=cv2.contourArea)
            x0, y0, bw, bh = cv2.boundingRect(contour)
            fruits.append({
                "contour": contour,
                "bbox": (int(x0), int(y0), int(bw), int(bh)),
                "area": int(area),
                "mask": comp_mask,
            })
    else:
        fruits = find_instances_auto(bin_mask, img_bgr, min_area, use_open, ws_fg_frac, ws_bg_dilate)

    out = img_bgr.copy()
    objects = []
    dets_for_panel = []

    for i, obj in enumerate(fruits, start=1):
        contour = obj["contour"]
        x0, y0, bw, bh = obj["bbox"]
        comp = obj["mask"]

        core = make_core_mask(comp, fruit_prob_full, core_erosion, core_prob_thr, core_min_px)

        qual_id, qual_conf, probs = mean_logits_on_mask(qual_logits_full, core)
        p_rotten = float(probs[1]) if len(probs) >= 2 else float(qual_conf)
        fruit_conf = float(fruit_prob_full[core].mean()) if core.any() else float(fruit_prob_full[comp].mean())


        is_rotten = (p_rotten >= float(rotten_score_thr))
        if only_rotten and (not is_rotten):
            continue


        if float(qual_conf) < float(cls_thr):
            qname = "unknown"
        else:
            if is_rotten and ("rotten" in [n.lower() for n in quality_names]):
                qname = "rotten"
            else:

                qname = quality_names[qual_id] if 0 <= qual_id < len(quality_names) else str(qual_id)

        conf = float(fruit_conf * max(qual_conf, p_rotten))

        color = color_for_label(qname)
        cv2.drawContours(out, [contour], -1, color, 2)

        label = f"{qname}, conf={conf:.2f} (fruit={fruit_conf:.2f}, pR={p_rotten:.2f})"

        ty = max(18, y0 - 6)
        cv2.rectangle(out, (x0, ty - 16), (x0 + min(420, W - x0 - 1), ty + 6), (0, 0, 0), -1)
        cv2.putText(out, label, (x0, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cx, cy = center_of_contour(contour)

        obj_rec = {
            "idx": int(i),
            "bbox_xywh": [int(x0), int(y0), int(bw), int(bh)],
            "quality": qname,
            "conf": conf,
            "fruit_conf": fruit_conf,
            "quality_conf": float(qual_conf),
            "p_rotten": p_rotten,
            "area": int(obj["area"]),
            "core_px": int(core.sum()),
            "center_xy": [float(cx), float(cy)],
        }
        objects.append(obj_rec)

        dets_for_panel.append({
            "idx": int(i),
            "quality": qname,
            "conf": float(conf),
            "fruit_conf": float(fruit_conf),
            "p_rotten": float(p_rotten),
            "center_xy": (float(cx), float(cy)),
            "color": color,
        })


    dets_for_panel.sort(key=lambda d: float(d["conf"]), reverse=True)

    out_panel = draw_side_panel(out, dets_for_panel) if panel else out


    p_rotten_map = None
    if qual_logits_full.shape[0] >= 2:

        a = qual_logits_full[0].astype(np.float32)
        b = qual_logits_full[1].astype(np.float32)
        m = np.maximum(a, b)
        ea = np.exp(a - m)
        eb = np.exp(b - m)
        p_rotten_map = (eb / (ea + eb + 1e-9)).astype(np.float32)

    return out_panel, bin_mask, fruit_prob_full, p_rotten_map, objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to best.pth from train_A.py")
    ap.add_argument("--input", required=True, help="Image path or folder")
    ap.add_argument("--output", required=True, help="Output image path or folder")

    ap.add_argument("--fruit-thr", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=700)

    ap.add_argument("--split", choices=["auto", "cc", "watershed"], default="auto")
    ap.add_argument("--ws-fg-frac", type=float, default=0.45)
    ap.add_argument("--ws-bg-dilate", type=int, default=2)

    ap.add_argument("--cls-thr", type=float, default=0.35)

    ap.add_argument("--only-rotten", action="store_true", help="Draw/save only instances predicted rotten.")
    ap.add_argument("--rotten-score-thr", type=float, default=0.50, help="Threshold on p(rotten) for rotten decision.")

    ap.add_argument("--no-open", action="store_true")

    ap.add_argument("--core-erosion", type=int, default=5)
    ap.add_argument("--core-prob-thr", type=float, default=0.70)
    ap.add_argument("--core-min-px", type=int, default=200)

    ap.add_argument("--panel", action="store_true", help="Add right-side detections panel.")

    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0|mps")
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--tf32", action="store_true")

    ap.add_argument("--save-mask", action="store_true")
    ap.add_argument("--save-prob", action="store_true", help="Save fruit_prob and p_rotten maps as images.")
    ap.add_argument("--write-json", action="store_true")
    args = ap.parse_args()


    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    device_type = get_device_type(device)

    if device_type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    ckpt = torch.load(args.weights, map_location="cpu")

    quality_names = ckpt.get("quality_names", ["fresh", "rotten"])
    if not isinstance(quality_names, (list, tuple)) or len(quality_names) < 2:
        quality_names = ["fresh", "rotten"]

    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        c = ckpt["cfg"]
        img_size = int(c.get("img_size", 512))
        base_c = int(c.get("base_c", 48))
    else:
        img_size = 512
        base_c = 48

    model = UNetRottenOnly(base_c=base_c, num_qualities=len(quality_names), head_drop=0.10).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    if device_type == "cuda" and args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    use_amp = bool(args.use_amp and AMP_AVAILABLE and device_type == "cuda")

    in_files = list_images(args.input)
    if len(in_files) == 0:
        raise FileNotFoundError(f"No images found in: {args.input}")

    out_is_dir = os.path.isdir(args.input) or args.output.endswith(os.sep) or (len(in_files) > 1)
    if out_is_dir:
        os.makedirs(args.output, exist_ok=True)

    for in_path in in_files:
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(in_path)

        out_img, mask, fruit_prob, p_rotten_map, objects = infer_one(
            model=model,
            img_bgr=img,
            img_size=img_size,
            quality_names=quality_names,
            device=device,
            use_amp=use_amp,
            channels_last=args.channels_last,
            fruit_thr=float(args.fruit_thr),
            min_area=int(args.min_area),
            split=str(args.split),
            ws_fg_frac=float(args.ws_fg_frac),
            ws_bg_dilate=int(args.ws_bg_dilate),
            cls_thr=float(args.cls_thr),
            only_rotten=bool(args.only_rotten),
            rotten_score_thr=float(args.rotten_score_thr),
            use_open=(not args.no_open),
            core_erosion=int(args.core_erosion),
            core_prob_thr=float(args.core_prob_thr),
            core_min_px=int(args.core_min_px),
            panel=bool(args.panel),
        )

        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(args.output, base + "_out.png") if out_is_dir else args.output
        cv2.imwrite(out_path, out_img)

        if args.save_mask:
            mask_path = os.path.join(os.path.dirname(out_path), base + "_mask.png") if out_is_dir else (os.path.splitext(out_path)[0] + "_mask.png")
            cv2.imwrite(mask_path, mask)

        if args.save_prob:
            fp = (np.clip(fruit_prob, 0, 1) * 255).astype(np.uint8)
            fp_path = os.path.join(os.path.dirname(out_path), base + "_fruitprob.png") if out_is_dir else (os.path.splitext(out_path)[0] + "_fruitprob.png")
            cv2.imwrite(fp_path, fp)

            if p_rotten_map is not None:
                pr = (np.clip(p_rotten_map, 0, 1) * 255).astype(np.uint8)
                pr_path = os.path.join(os.path.dirname(out_path), base + "_protten.png") if out_is_dir else (os.path.splitext(out_path)[0] + "_protten.png")
                cv2.imwrite(pr_path, pr)

        if args.write_json:
            det_path = os.path.join(os.path.dirname(out_path), base + "_det.json") if out_is_dir else (os.path.splitext(out_path)[0] + "_det.json")
            with open(det_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"image": os.path.basename(in_path), "quality_names": list(quality_names), "objects": objects},
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        if not out_is_dir:
            break

    print("Готово.")


if __name__ == "__main__":
    main()