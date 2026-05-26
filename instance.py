import json
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast


@dataclass
class Config:
    weights_path: str = r"Train_Unet/runs/weights/best_model.pth"
    coco_ann_path: str = r"Train_Unet/annotations_mosaic.json"
    input_video: str = r"Train_Unet/input.mp4"
    output_video: str = r"Train_Unet/output_unet_overlay_3.mp4"

    img_size: int = 512
    base_c: int = 64
    use_amp: bool = True

    prob_threshold: float = 0.9
    min_region_area: int = 500

    overlay_alpha: float = 0.5
    font_scale: float = 0.6
    font_thickness: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def build_trainid_to_name(ann_path: str) -> Dict[int, str]:
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    categories = coco["categories"]
    id_to_name = {c["id"]: c["name"] for c in categories}
    cat_ids = sorted(id_to_name.keys())
    trainid_to_name = {i + 1: id_to_name[cid] for i, cid in enumerate(cat_ids)}
    return trainid_to_name


def make_color_palette(num_classes: int) -> np.ndarray:
    rng = np.random.RandomState(0)
    palette = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def mask_to_color(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = palette[mask.reshape(-1)].reshape(h, w, 3)
    return color


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int, base_c: int = 64):
        super().__init__()
        self.inc = DoubleConv(3, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 8)
        self.up1 = Up(base_c * 16, base_c * 4)
        self.up2 = Up(base_c * 8, base_c * 2)
        self.up3 = Up(base_c * 4, base_c)
        self.up4 = Up(base_c * 2, base_c)
        self.outc = OutConv(base_c, num_classes)

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
        logits = self.outc(x)
        return logits


def load_model(cfg: Config) -> Tuple[UNet, np.ndarray, Dict[int, str]]:
    trainid_to_name = build_trainid_to_name(cfg.coco_ann_path)
    num_classes = len(trainid_to_name) + 1
    model = UNet(num_classes=num_classes, base_c=cfg.base_c)
    state = torch.load(
        cfg.weights_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()

    palette = make_color_palette(num_classes)
    return model, palette, trainid_to_name


def process_video(cfg: Config):
    model, palette, id2name = load_model(cfg)

    cap = cv2.VideoCapture(cfg.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(cfg.output_video, fourcc, fps, (width, height))

    device = cfg.device
    use_amp = cfg.use_amp

    scale_x = width / cfg.img_size
    scale_y = height / cfg.img_size

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            img_resized = cv2.resize(
                frame_rgb, (cfg.img_size, cfg.img_size),
                interpolation=cv2.INTER_LINEAR,
            )
            img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            with autocast(device_type="cuda", enabled=(use_amp and device == "cuda")):
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)

            probs = probs[0]
            conf, pred_t = torch.max(probs, dim=0)
            pred = pred_t.cpu().numpy().astype(np.uint8)
            conf = conf.cpu().numpy()

            pred[conf < cfg.prob_threshold] = 0

            mask_fg = (pred != 0).astype(np.uint8)
            num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
                mask_fg, connectivity=8
            )

            regions = []
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area < cfg.min_region_area:
                    pred[labels_cc == label] = 0
                    continue

                region_pixels = pred[labels_cc == label]
                if region_pixels.size == 0:
                    continue

                cls_id = int(np.bincount(region_pixels).argmax())
                if cls_id == 0:
                    continue

                cx, cy = centroids[label]
                regions.append((cx, cy, cls_id))

            mask_color = mask_to_color(pred, palette)
            mask_color = cv2.resize(
                mask_color, (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            overlay_rgb = cv2.addWeighted(
                frame_rgb, 1.0 - cfg.overlay_alpha,
                mask_color, cfg.overlay_alpha,
                0.0,
            )

            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            for cx, cy, cls_id in regions:
                x = int(cx * scale_x)
                y = int(cy * scale_y)
                name = id2name.get(cls_id, str(cls_id))

                cv2.putText(
                    overlay_bgr,
                    name,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    cfg.font_scale,
                    (0, 0, 0),
                    thickness=cfg.font_thickness + 2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    overlay_bgr,
                    name,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    cfg.font_scale,
                    (255, 255, 255),
                    thickness=cfg.font_thickness,
                    lineType=cv2.LINE_AA,
                )

            out.write(overlay_bgr)

    cap.release()
    out.release()


if __name__ == "__main__":
    process_video(cfg)
