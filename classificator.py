import os
import json
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class Config:
    coco_ann_path: str = r"D:\YOLOFRUIT\dataset_generator\synthetic_keep_mapping\annotations.json"
    img_dir: str = r"D:\YOLOFRUIT\dataset_generator\synthetic_keep_mapping"
    crops_root: str = r"dataset_generator/cls_fruit_quality"
    label_mode: str = "quality"
    bbox_expand: float = 1.2
    min_size: int = 32
    prepare_crops: bool = True
    output_dir: str = "runs_fruit_cls"
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-5
    val_split: float = 0.2
    use_amp: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(os.path.join(cfg.output_dir, "weights"), exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_coco(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco


def infer_quality_from_name(name: str) -> str:
    name_l = name.lower()
    if "fresh" in name_l:
        return "fresh"
    if "rotten" in name_l:
        return "rotten"
    parts = name_l.split("_")
    if parts and parts[0] in ("fresh", "rotten"):
        return parts[0]
    return ""


def export_crops_for_classifier(cfg: Config):
    coco = load_coco(cfg.coco_ann_path)
    images = {img["id"]: img for img in coco["images"]}
    categories = coco["categories"]
    anns = coco["annotations"]

    catid_to_name: Dict[int, str] = {c["id"]: c["name"] for c in categories}

    if os.path.exists(cfg.crops_root) and not cfg.prepare_crops:
        return

    if cfg.prepare_crops and os.path.exists(cfg.crops_root):
        import shutil
        shutil.rmtree(cfg.crops_root)

    os.makedirs(cfg.crops_root, exist_ok=True)

    for ann in tqdm(anns, desc="Exporting crops"):
        cid = ann["category_id"]
        if cid not in catid_to_name:
            continue

        name = catid_to_name[cid]
        quality = infer_quality_from_name(name)
        if quality == "":
            continue

        if cfg.label_mode == "quality":
            label = quality
        else:
            label = name.lower()

        img_info = images.get(ann["image_id"])
        if img_info is None:
            continue

        img_path = os.path.join(cfg.img_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        h0, w0 = img.shape[:2]

        x, y, w, h = ann["bbox"]
        cx = x + w / 2.0
        cy = y + h / 2.0
        w2 = w * cfg.bbox_expand
        h2 = h * cfg.bbox_expand

        x1 = max(0, int(cx - w2 / 2.0))
        y1 = max(0, int(cy - h2 / 2.0))
        x2 = min(w0, int(cx + w2 / 2.0))
        y2 = min(h0, int(cy + h2 / 2.0))

        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) < cfg.min_size or (y2 - y1) < cfg.min_size:
            continue

        crop = img[y1:y2, x1:x2]
        label_dir = os.path.join(cfg.crops_root, label)
        os.makedirs(label_dir, exist_ok=True)

        out_name = "%s_%06d.jpg" % (label, ann["id"])
        out_path = os.path.join(label_dir, out_name)
        cv2.imwrite(out_path, crop)


class SubsetWithTransform(Dataset):
    def __init__(self, base_dataset: datasets.ImageFolder, indices: List[int], transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, cls = self.base_dataset.samples[real_idx]
        img = self.base_dataset.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, cls


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, List[str]]:
    base_dataset = datasets.ImageFolder(root=cfg.crops_root, transform=None)
    class_names = base_dataset.classes

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    n_total = len(indices)
    n_val = int(n_total * cfg.val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = SubsetWithTransform(base_dataset, train_indices, train_tf)
    val_ds = SubsetWithTransform(base_dataset, val_indices, val_tf)

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

    return train_loader, val_loader, class_names


def build_model(num_classes: int, device: str):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)
    return model


def train_classifier(cfg: Config):
    set_seed(cfg.seed)

    train_loader, val_loader, class_names = build_dataloaders(cfg)
    num_classes = len(class_names)

    device = cfg.device
    model = build_model(num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler() if cfg.use_amp and device == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc="Train", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if cfg.use_amp and scaler is not None and device == "cuda":
                with autocast(device_type="cuda", enabled=True):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

        train_loss = running_loss / max(1, total_samples)

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val", leave=False):
                imgs = imgs.to(device)
                labels = labels.to(device)

                with autocast(device_type="cuda", enabled=(cfg.use_amp and device == "cuda")):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                val_loss_total += loss.item() * imgs.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_loss = val_loss_total / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        epoch_time = time.perf_counter() - epoch_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(cfg.output_dir, "weights", "best_cls.pth")
            torch.save(
                {"model_state_dict": model.state_dict(), "class_names": class_names, "cfg": cfg.__dict__},
                best_path,
            )

        plot_curves(history, cfg)


def plot_curves(history: Dict[str, List[float]], cfg: Config):
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
    plt.plot(epochs, history["val_acc"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "training_curves_cls.png"))
    plt.close()


def main():
    set_seed(cfg.seed)
    export_crops_for_classifier(cfg)
    train_classifier(cfg)


if __name__ == "__main__":
    main()