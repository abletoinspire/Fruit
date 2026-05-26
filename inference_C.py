# python inference_C.py --weights .\runs_C_condqual\weights\best.pth --input .\test_images --output .\out_C --fruit-thr 0.55 --min-area 700 --split auto --core-erosion 5 --core-prob-thr 0.70 --core-min-px 200 --only-rotten --rotten-score-thr 0.55 --panel --save-maps
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


class UNetConditionalQuality(nn.Module):

    def __init__(self, base_c: int, num_types: int = 3, num_qualities: int = 2, head_drop: float = 0.10):
        super().__init__()
        assert num_types == 3
        assert num_qualities == 2

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

        self.head_quality_experts = nn.ModuleList([
            SegClassHead(c1, num_qualities, drop_p=head_drop),
            SegClassHead(c1, num_qualities, drop_p=head_drop),
            SegClassHead(c1, num_qualities, drop_p=head_drop),
        ])

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

        fruit_logits = self.head_fruit(x)
        type_logits = self.head_type(x)
        qual_logits = torch.stack([h(x) for h in self.head_quality_experts], dim=1)
        return fruit_logits, type_logits, qual_logits


def get_device_type(device: str) -> str:
    d = str(device).lower()
    if d.startswith("cuda"):
        return "cuda"
    if d == "mps":
        return "mps"
    return "cpu"


def list_images(path: str):
    if os.path.isdir(path):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(exts)]
    return [path]


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
        value=(pad_value, pad_value, pad_value),
    )
    meta = {"top": int(top), "left": int(left), "new_h": int(nh), "new_w": int(nw)}
    return out, meta


def unletterbox_map(map_s: np.ndarray, meta: dict, H: int, W: int, is_mask: bool):
    top, left = meta["top"], meta["left"]
    nh, nw = meta["new_h"], meta["new_w"]
    crop = map_s[top:top + nh, left:left + nw]
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(crop, (W, H), interpolation=interp)


def softmax_1d(x: np.ndarray):
    x = x.astype(np.float32)
    x = x - float(x.max())
    e = np.exp(x)
    return e / (float(e.sum()) + 1e-9)


def mean_logits_on_mask(logits_chw: np.ndarray, mask_bool: np.ndarray):
    if not mask_bool.any():
        pooled = logits_chw.reshape(logits_chw.shape[0], -1).mean(axis=1)
    else:
        pooled = logits_chw[:, mask_bool].mean(axis=1)
    probs = softmax_1d(pooled)
    cls_id = int(np.argmax(probs))
    return cls_id, float(probs[cls_id]), probs


def make_core_mask(comp_mask: np.ndarray, fruit_prob: np.ndarray, erosion: int, prob_thr: float, core_min_px: int):
    if erosion <= 0:
        core = comp_mask.copy()
    else:
        u8 = comp_mask.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion))
        er = cv2.erode(u8, k, iterations=1)
        core = (er > 0)

    core = core & (fruit_prob >= float(prob_thr))

    if int(core.sum()) < int(core_min_px):
        core2 = comp_mask.copy() & (fruit_prob >= float(prob_thr))
        if int(core2.sum()) >= int(core_min_px):
            return core2
        return comp_mask

    return core


def center_of_contour(contour):
    m = cv2.moments(contour)
    if abs(m.get("m00", 0.0)) < 1e-6:
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w * 0.5, y + h * 0.5)
    return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))


def color_for_pair(type_name: str, quality_name: str):
    t = (type_name or "").lower()
    q = (quality_name or "").lower()
    if "apple" in t and "fresh" in q:
        return (255, 180, 0)
    if "apple" in t and "rotten" in q:
        return (0, 140, 255)
    if "banana" in t and "fresh" in q:
        return (80, 220, 80)
    if "banana" in t and "rotten" in q:
        return (80, 80, 255)
    if "orange" in t and "fresh" in q:
        return (220, 80, 180)
    if "orange" in t and "rotten" in q:
        return (40, 160, 255)
    return (200, 200, 200)


def draw_side_panel(base_bgr: np.ndarray, dets: list):
    H, W = base_bgr.shape[:2]
    panel_w = max(620, int(W * 0.48))
    out = np.zeros((H, W + panel_w, 3), dtype=np.uint8)
    out[:, :W] = base_bgr
    out[:, W:] = (30, 30, 30)

    cv2.putText(out, f"Detections: {len(dets)}", (W + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    y0 = 80
    line_h = 38
    max_lines = max(1, (H - y0 - 20) // line_h)
    dets_show = dets[:max_lines]

    for i, d in enumerate(dets_show):
        y = y0 + i * line_h
        cv2.rectangle(out, (W + 20, y - 16), (W + 36, y), d["color"], -1)
        text = (
            f"#{d['idx']}  {d['type']}  {d['quality']}  conf={d['conf']:.2f}  "
            f"(fruit={d['fruit_conf']:.2f}, t={d['type_conf']:.2f}, pR={d['p_rotten']:.2f})"
        )
        cv2.putText(out, text, (W + 50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 235, 235), 2, cv2.LINE_AA)

        cx, cy = d["center_xy"]
        tx, ty = W + 20, y - 8
        cv2.line(out, (int(cx), int(cy)), (int(tx), int(ty)), (80, 160, 80), 1, cv2.LINE_AA)

    if len(dets) > len(dets_show):
        cv2.putText(out, f"... +{len(dets) - len(dets_show)} more", (W + 20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
    return out


def softmax2_to_protten(logit0: np.ndarray, logit1: np.ndarray) -> np.ndarray:
    a = logit0.astype(np.float32)
    b = logit1.astype(np.float32)
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    return (eb / (ea + eb + 1e-9)).astype(np.float32)


def type_softmax_maps(type_logits_full: np.ndarray):

    m = np.max(type_logits_full, axis=0)
    exps = np.exp(type_logits_full - m[None, :, :])
    denom = np.sum(exps, axis=0) + 1e-9
    probs = exps / denom
    type_pred = np.argmax(probs, axis=0).astype(np.uint8)
    type_conf = np.max(probs, axis=0).astype(np.float32)
    return type_pred, type_conf


def _filter_seed_components(seed_u8: np.ndarray, min_seed_area: int = 40):

    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed_u8.astype(np.uint8), connectivity=8)
    if n <= 1:
        return labels.astype(np.int32), 0

    out = np.zeros_like(labels, dtype=np.int32)
    keep_id = 1
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_seed_area):
            continue
        out[labels == i] = keep_id
        keep_id += 1

    return out, keep_id - 1


def split_component_adaptive(
    roi_img_bgr: np.ndarray,
    roi_prob: np.ndarray,
    roi_lo: np.ndarray,
    fruit_thr_hi_start: float,
    min_area: int,
    hi_erode: int = 1,
    min_seed_area: int = 40,
):


    thr = float(fruit_thr_hi_start)
    thr_max = 0.95
    step = 0.03

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    best_labels = None
    best_seeds = 0

    while thr <= thr_max:
        roi_hi = ((roi_prob >= thr) & (roi_lo > 0)).astype(np.uint8)

        if hi_erode > 0:
            roi_hi = cv2.erode(roi_hi, k3, iterations=int(hi_erode))

        seed_labels, seeds = _filter_seed_components(roi_hi, min_seed_area=min_seed_area)
        if seeds >= 2:
            best_labels = seed_labels
            best_seeds = seeds
            break

        thr += step


    if best_seeds < 2:
        dist = cv2.distanceTransform(roi_lo.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
        if float(dist.max()) > 1e-6:
            dil = cv2.dilate(dist, k3)
            peaks = (dist == dil) & (dist >= max(2.0, float(dist.max()) * 0.10))
            peak_u8 = peaks.astype(np.uint8)

            seed_labels, seeds = _filter_seed_components(peak_u8, min_seed_area=1)
            if seeds >= 2:
                best_labels = seed_labels
                best_seeds = seeds

    if best_seeds < 2 or best_labels is None:
        return [(roi_lo > 0)]


    markers = np.zeros(roi_lo.shape[:2], dtype=np.int32)
    markers[roi_lo == 0] = 1
    markers[best_labels > 0] = best_labels[best_labels > 0] + 1

    ws = cv2.watershed(roi_img_bgr, markers)

    parts = []
    for mid in range(2, 2 + best_seeds):
        m = (ws == mid) & (roi_lo > 0)
        if int(m.sum()) >= int(min_area):
            parts.append(m)


    if len(parts) <= 1:
        return [(roi_lo > 0)]

    return parts


def instances_from_fruit_prob(
    img_bgr: np.ndarray,
    fruit_prob: np.ndarray,
    fruit_thr: float,
    fruit_thr_hi: float,
    min_area: int,
    use_open: bool,
    open_ksize: int,
    hi_erode: int = 1,
    min_seed_area: int = 40,
):

    H, W = fruit_prob.shape[:2]

    lo = (fruit_prob >= float(fruit_thr)).astype(np.uint8)

    if use_open and open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_ksize), int(open_ksize)))
        lo = cv2.morphologyEx(lo, cv2.MORPH_OPEN, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(lo, connectivity=8)
    instances = []

    for lbl in range(1, num):
        x, y, w, h, area = stats[lbl]
        if int(area) < int(min_area):
            continue

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        comp = (labels == lbl)
        roi_img = img_bgr[y1:y2, x1:x2].copy()
        roi_lo = comp[y1:y2, x1:x2].astype(np.uint8)
        roi_prob = fruit_prob[y1:y2, x1:x2].astype(np.float32)

        roi_parts = split_component_adaptive(
            roi_img_bgr=roi_img,
            roi_prob=roi_prob,
            roi_lo=roi_lo,
            fruit_thr_hi_start=float(fruit_thr_hi),
            min_area=int(min_area),
            hi_erode=int(hi_erode),
            min_seed_area=int(min_seed_area),
        )

        for part in roi_parts:
            full = np.zeros((H, W), dtype=bool)
            full[y1:y2, x1:x2] = part

            a = int(full.sum())
            if a < int(min_area):
                continue

            u8 = full.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            contour = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(contour)

            instances.append({
                "mask": full,
                "contour": contour,
                "bbox": (int(bx), int(by), int(bw), int(bh)),
                "area": a,
            })

    return instances


def infer_one(
    model: nn.Module,
    img_bgr: np.ndarray,
    img_size: int,
    type_names: list,
    quality_names: list,
    device: str,
    use_amp: bool,
    channels_last: bool,
    fruit_thr: float,
    fruit_thr_hi: float,
    min_area: int,
    only_rotten: bool,
    rotten_score_thr: float,
    core_erosion: int,
    core_prob_thr: float,
    core_min_px: int,
    panel: bool,
    save_maps: bool,
    use_open: bool,
    open_ksize: int,
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
                fruit_logits, type_logits, qual_logits = model(x)
        else:
            fruit_logits, type_logits, qual_logits = model(x)

        fruit_prob_s = torch.sigmoid(fruit_logits)[0, 0]
        type_logits_s = type_logits[0]
        qual_logits_s = qual_logits[0]

    fruit_prob_np = fruit_prob_s.float().cpu().numpy()
    type_logits_np = type_logits_s.float().cpu().numpy()
    qual_logits_np = qual_logits_s.float().cpu().numpy()

    fruit_prob = unletterbox_map(fruit_prob_np, meta, H, W, is_mask=False).astype(np.float32)
    type_logits_full = np.stack(
        [unletterbox_map(type_logits_np[c], meta, H, W, is_mask=False) for c in range(3)],
        axis=0
    ).astype(np.float32)

    qual_logits_full = np.zeros((3, 2, H, W), dtype=np.float32)
    for t in range(3):
        for q in range(2):
            qual_logits_full[t, q] = unletterbox_map(qual_logits_np[t, q], meta, H, W, is_mask=False).astype(np.float32)


    instances = instances_from_fruit_prob(
        img_bgr=img_bgr,
        fruit_prob=fruit_prob,
        fruit_thr=fruit_thr,
        fruit_thr_hi=fruit_thr_hi,
        min_area=min_area,
        use_open=use_open,
        open_ksize=open_ksize,
    )

    out = img_bgr.copy()
    objects = []
    dets_panel = []
    idx_counter = 1

    for it in instances:
        comp = it["mask"]
        contour = it["contour"]
        x0, y0, bw, bh = it["bbox"]

        core = make_core_mask(comp, fruit_prob, core_erosion, core_prob_thr, core_min_px)
        fruit_conf = float(fruit_prob[core].mean()) if core.any() else float(fruit_prob[comp].mean())


        t_id, t_conf, _t_probs = mean_logits_on_mask(type_logits_full, core)
        type_name = type_names[t_id] if 0 <= t_id < len(type_names) else str(t_id)


        q_id, q_conf, q_probs = mean_logits_on_mask(qual_logits_full[t_id], core)
        quality_name = quality_names[q_id] if 0 <= q_id < len(quality_names) else str(q_id)

        p_rotten = float(q_probs[1]) if len(q_probs) >= 2 else float(q_conf)

        if only_rotten and (p_rotten < float(rotten_score_thr)):
            continue

        conf = float(fruit_conf * t_conf * max(q_conf, p_rotten))
        color = color_for_pair(type_name, quality_name)
        cv2.drawContours(out, [contour], -1, color, 2)

        label = f"{type_name}, {quality_name}, conf={conf:.2f} (fruit={fruit_conf:.2f}, t={t_conf:.2f}, pR={p_rotten:.2f})"
        ty = max(18, y0 - 6)
        cv2.rectangle(out, (x0, ty - 16), (x0 + min(560, W - x0 - 1), ty + 6), (0, 0, 0), -1)
        cv2.putText(out, label, (x0, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        cx, cy = center_of_contour(contour)

        objects.append({
            "idx": int(idx_counter),
            "bbox_xywh": [int(x0), int(y0), int(bw), int(bh)],
            "type": type_name,
            "type_conf": float(t_conf),
            "quality": quality_name,
            "quality_conf": float(q_conf),
            "p_rotten": float(p_rotten),
            "fruit_conf": float(fruit_conf),
            "conf": float(conf),
            "area": int(it["area"]),
            "core_px": int(core.sum()),
            "center_xy": [float(cx), float(cy)],
        })

        dets_panel.append({
            "idx": int(idx_counter),
            "type": type_name,
            "quality": quality_name,
            "conf": float(conf),
            "fruit_conf": float(fruit_conf),
            "type_conf": float(t_conf),
            "p_rotten": float(p_rotten),
            "center_xy": (float(cx), float(cy)),
            "color": color,
        })

        idx_counter += 1

    dets_panel.sort(key=lambda d: float(d["conf"]), reverse=True)
    out_panel = draw_side_panel(out, dets_panel) if panel else out


    type_pred_map, _type_conf_map = type_softmax_maps(type_logits_full)
    qual_pred_map = np.zeros((H, W), dtype=np.uint8)
    p_rotten_map = np.zeros((H, W), dtype=np.float32)
    for t in range(3):
        m = (type_pred_map == t)
        if not m.any():
            continue
        prott = softmax2_to_protten(qual_logits_full[t, 0], qual_logits_full[t, 1])
        p_rotten_map[m] = prott[m]
        qual_pred_map[m] = (qual_logits_full[t, 1] > qual_logits_full[t, 0]).astype(np.uint8)[m]


    bin_mask_u8 = (fruit_prob >= float(fruit_thr)).astype(np.uint8) * 255
    return out_panel, bin_mask_u8, fruit_prob, type_pred_map, qual_pred_map, p_rotten_map, objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--fruit-thr", type=float, default=0.55)
    ap.add_argument("--fruit-thr-hi", type=float, default=-1.0)
    ap.add_argument("--min-area", type=int, default=700)

    ap.add_argument("--split", choices=["auto", "cc", "watershed"], default="auto")
    ap.add_argument("--core-erosion", type=int, default=5)
    ap.add_argument("--core-prob-thr", type=float, default=0.70)
    ap.add_argument("--core-min-px", type=int, default=200)

    ap.add_argument("--only-rotten", action="store_true")
    ap.add_argument("--rotten-score-thr", type=float, default=0.55)

    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--write-json", action="store_true")
    ap.add_argument("--save-mask", action="store_true")
    ap.add_argument("--save-maps", action="store_true")

    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0|mps")
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--tf32", action="store_true")


    ap.add_argument("--open-ksize", type=int, default=3)
    ap.add_argument("--no-open", action="store_true")
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
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    img_size = int(cfg.get("img_size", 512))
    base_c = int(cfg.get("base_c", 48))

    type_names = ckpt.get("type_names", ["apple", "banana", "orange"]) if isinstance(ckpt, dict) else ["apple", "banana", "orange"]
    quality_names = ckpt.get("quality_names", ["fresh", "rotten"]) if isinstance(ckpt, dict) else ["fresh", "rotten"]

    model = UNetConditionalQuality(base_c=base_c, num_types=3, num_qualities=2, head_drop=0.10).to(device)
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

    pal_type = np.array([[255, 80, 80], [80, 255, 80], [80, 80, 255]], dtype=np.uint8)
    pal_qual = np.array([[80, 180, 255], [255, 200, 80]], dtype=np.uint8)

    fruit_thr = float(args.fruit_thr)
    fruit_thr_hi = float(args.fruit_thr_hi)
    if fruit_thr_hi < 0:
        fruit_thr_hi = min(0.95, fruit_thr + 0.12)

    use_open = (not args.no_open)
    open_ksize = int(args.open_ksize)

    for in_path in in_files:
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(in_path)

        out_img, mask_u8, fruit_prob, type_map, qual_map, p_rotten_map, objects = infer_one(
            model=model,
            img_bgr=img,
            img_size=img_size,
            type_names=list(type_names),
            quality_names=list(quality_names),
            device=device,
            use_amp=use_amp,
            channels_last=args.channels_last,
            fruit_thr=fruit_thr,
            fruit_thr_hi=fruit_thr_hi,
            min_area=int(args.min_area),
            only_rotten=bool(args.only_rotten),
            rotten_score_thr=float(args.rotten_score_thr),
            core_erosion=int(args.core_erosion),
            core_prob_thr=float(args.core_prob_thr),
            core_min_px=int(args.core_min_px),
            panel=bool(args.panel),
            save_maps=bool(args.save_maps),
            use_open=use_open,
            open_ksize=open_ksize,
        )

        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(args.output, base + "_out.png") if out_is_dir else args.output
        cv2.imwrite(out_path, out_img)

        if args.save_mask:
            mp = os.path.join(os.path.dirname(out_path), base + "_mask.png") if out_is_dir else (os.path.splitext(out_path)[0] + "_mask.png")
            cv2.imwrite(mp, mask_u8)

        if args.save_maps:
            fp = (np.clip(fruit_prob, 0, 1) * 255).astype(np.uint8)
            fp_path = os.path.join(os.path.dirname(out_path), base + "_fruitprob.png") if out_is_dir else (os.path.splitext(out_path)[0] + "_fruitprob.png")
            cv2.imwrite(fp_path, fp)

            tcol = pal_type[type_map.reshape(-1)].reshape(type_map.shape[0], type_map.shape[1], 3)
            t_path = os.path.join(os.path.dirname(out_path), base + "_typepred.png") if out_is_dir else (os.path.splitext(out_path)[0] + "_typepred.png")
            cv2.imwrite(t_path, cv2.cvtColor(tcol, cv2.COLOR_RGB2BGR))

            qcol = pal_qual[qual_map.reshape(-1)].reshape(qual_map.shape[0], qual_map.shape[1], 3)
            q_path = os.path.join(os.path.dirname(out_path), base + "_qualpred.png") if out_is_dir else (os.path.splitext(out_path)[0] + "_qualpred.png")
            cv2.imwrite(q_path, cv2.cvtColor(qcol, cv2.COLOR_RGB2BGR))

            pr = (np.clip(p_rotten_map, 0, 1) * 255).astype(np.uint8)
            pr_path = os.path.join(os.path.dirname(out_path), base + "_protten.png") if out_is_dir else (os.path.splitext(out_path)[0] + "_protten.png")
            cv2.imwrite(pr_path, pr)

        if args.write_json:
            jp = os.path.join(os.path.dirname(out_path), base + "_det.json") if out_is_dir else (os.path.splitext(out_path)[0] + "_det.json")
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(
                    {"image": os.path.basename(in_path), "type_names": list(type_names), "quality_names": list(quality_names), "objects": objects},
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        if not out_is_dir:
            break

    print(f"Готово. Использованный fruit_thr_hi: {fruit_thr_hi:.2f}")


if __name__ == "__main__":
    main()
