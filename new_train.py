import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.cuda.amp import GradScaler
from torch.amp import autocast

@dataclass
class Config:

    img_dir: str = r"dataset_generator/synthetic"
    ann_path: str = r"dataset_generator/synthetic/annotations.json"
    output_dir: str = r"dataset_generator/runs_new"
    mask_dir: str = r"dataset_generator/masks_cache"

    img_size: int = 512
    batch_size: int = 12
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    val_split: float = 0.2

    base_c: int = 64
    use_amp: bool = True

    vis_every: int = 5

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(os.path.join(cfg.output_dir, "vis"), exist_ok=True)
os.makedirs(os.path.join(cfg.output_dir, "weights"), exist_ok=True)
os.makedirs(cfg.mask_dir, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_coco_index_fruit_types(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    imgid_to_img = {img["id"]: img for img in images}
    imgid_to_anns: Dict[int, List[dict]] = {}
    for ann in annotations:
        imgid_to_anns.setdefault(ann["image_id"], []).append(ann)

    catid_to_name = {c["id"]: c["name"] for c in categories}

    catid_to_fruitname: Dict[int, str] = {}
    fruitnames_set = set()
    for cid, name in catid_to_name.items():
        parts = name.split("_")
        fruit = parts[-1] if len(parts) > 1 else name
        fruit = fruit.lower()
        catid_to_fruitname[cid] = fruit
        fruitnames_set.add(fruit)

    fruitnames_sorted = sorted(fruitnames_set)
    fruitname_to_trainid = {fruit: i + 1 for i, fruit in enumerate(fruitnames_sorted)}
    trainid_to_fruitname = {i + 1: fruit for i, fruit in enumerate(fruitnames_sorted)}

    catid_to_trainid = {
        cid: fruitname_to_trainid[fruit]
        for cid, fruit in catid_to_fruitname.items()
    }

    return (
        coco,
        imgid_to_img,
        imgid_to_anns,
        catid_to_trainid,
        trainid_to_fruitname,
    )

def polygons_to_mask(
    anns: List[dict],
    catid_to_trainid: Dict[int, int],
    h: int,
    w: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    for ann in anns:
        if ann.get("iscrowd", 0) == 1:
            continue
        cat_id = ann["category_id"]
        if cat_id not in catid_to_trainid:
            continue

        class_id = catid_to_trainid[cat_id]

        seg = ann.get("segmentation", [])
        if not seg:
            continue
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts_int = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts_int], class_id)

    return mask

class FruitSegDataset(Dataset):

    def __init__(self, cfg: Config, indices=None, is_train: bool = True):
        self.cfg = cfg
        self.is_train = is_train

        (
            coco,
            imgid_to_img,
            imgid_to_anns,
            catid_to_trainid,
            trainid_to_fruitname,
        ) = build_coco_index_fruit_types(cfg.ann_path)

        self.coco = coco
        self.imgid_to_img = imgid_to_img
        self.imgid_to_anns = imgid_to_anns
        self.catid_to_trainid = catid_to_trainid
        self.trainid_to_fruitname = trainid_to_fruitname

        self.images = list(imgid_to_img.values())
        if indices is None:
            self.indices = list(range(len(self.images)))
        else:
            self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_info = self.images[real_idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.cfg.img_dir, file_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        mask_name = os.path.splitext(file_name)[0] + ".png"
        mask_path = os.path.join(self.cfg.mask_dir, mask_name)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape[:2] != (h0, w0):
                anns = self.imgid_to_anns.get(img_id, [])
                mask = polygons_to_mask(anns, self.catid_to_trainid, h0, w0)
                cv2.imwrite(mask_path, mask)
        else:
            anns = self.imgid_to_anns.get(img_id, [])
            mask = polygons_to_mask(anns, self.catid_to_trainid, h0, w0)
            cv2.imwrite(mask_path, mask)

        img_resized = cv2.resize(
            img, (self.cfg.img_size, self.cfg.img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_resized = cv2.resize(
            mask, (self.cfg.img_size, self.cfg.img_size),
            interpolation=cv2.INTER_NEAREST,
        )

        if self.is_train and random.random() < 0.5:
            img_resized = np.ascontiguousarray(np.fliplr(img_resized))
            mask_resized = np.ascontiguousarray(np.fliplr(mask_resized))

        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask_resized).long()

        return img_tensor, mask_tensor, file_name

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

def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(1, num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        if target_inds.sum() == 0:
            continue
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0

def make_color_palette(num_classes: int) -> np.ndarray:
    rng = np.random.RandomState(0)
    palette = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette

def mask_to_color(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = palette[mask.reshape(-1)].reshape(h, w, 3)
    return color

def visualize_predictions(
    model: nn.Module,
    loader: DataLoader,
    palette: np.ndarray,
    epoch: int,
    cfg: Config,
    num_samples: int = 3,
):
    model.eval()
    device = cfg.device
    samples_done = 0

    with torch.no_grad():
        for imgs, masks, names in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            with autocast(device_type="cuda", enabled=(cfg.use_amp and device == "cuda")):
                logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            imgs_np = (imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            masks_np = masks.cpu().numpy().astype(np.uint8)
            preds_np = preds.cpu().numpy().astype(np.uint8)

            for img_np, gt_np, pr_np, fname in zip(imgs_np, masks_np, preds_np, names):
                gt_color = mask_to_color(gt_np, palette)
                pr_color = mask_to_color(pr_np, palette)

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(img_np)
                ax[0].set_title("Image")
                ax[1].imshow(gt_color)
                ax[1].set_title("GT fruit type")
                ax[2].imshow(pr_color)
                ax[2].set_title("Pred fruit type")
                for a in ax:
                    a.axis("off")

                out_path = os.path.join(
                    cfg.output_dir,
                    "vis",
                    f"epoch_{epoch:03d}_{os.path.basename(fname)}.png",
                )
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close(fig)

                samples_done += 1
                if samples_done >= num_samples:
                    return

def plot_curves(history: dict, cfg: Config):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_iou"], label="val mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "training_curves.png"))
    plt.close()

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    scaler: GradScaler,
    use_amp: bool,
):
    model.train()
    running_loss = 0.0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None and device == "cuda":
            with autocast(device_type="cuda", enabled=True):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: str,
    num_classes: int,
    use_amp: bool,
):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    n_batches = 0

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            with autocast(device_type="cuda", enabled=(use_amp and device == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, masks)

            running_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(logits, dim=1)
            iou = compute_iou(preds, masks, num_classes)
            running_iou += iou
            n_batches += 1

    epoch_loss = running_loss / len(loader.dataset)
    mean_iou = running_iou / max(1, n_batches)
    return epoch_loss, mean_iou

def main():
    set_seed(cfg.seed)

    base_ds = FruitSegDataset(cfg, indices=None, is_train=True)
    num_classes = len(base_ds.trainid_to_fruitname) + 1
    n_total = len(base_ds.images)

    indices = list(range(n_total))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(indices)
    n_val = int(n_total * cfg.val_split)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = FruitSegDataset(cfg, indices=train_idx, is_train=True)
    val_ds = FruitSegDataset(cfg, indices=val_idx, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    device = cfg.device
    model = UNet(num_classes=num_classes, base_c=cfg.base_c).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler() if cfg.use_amp and device == "cuda" else None

    palette = make_color_palette(num_classes)

    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_val_iou = 0.0

    for epoch in range(1, cfg.epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            use_amp=cfg.use_amp,
        )
        val_loss, val_iou = validate(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            use_amp=cfg.use_amp,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        if (
            epoch == 1
            or epoch == cfg.epochs
            or (epoch % cfg.vis_every == 0)
        ):
            visualize_predictions(
                model, val_loader, palette, epoch, cfg, num_samples=3
            )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = os.path.join(cfg.output_dir, "weights", "best_fruit_seg.pth")
            torch.save(model.state_dict(), best_path)

        ckpt_path = os.path.join(
            cfg.output_dir, "weights", f"epoch_{epoch:03d}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)

        plot_curves(history, cfg)

if __name__ == "__main__":
    main()
