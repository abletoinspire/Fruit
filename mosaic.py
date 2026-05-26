import json
import os
import random
from copy import deepcopy

import cv2
import numpy as np

IMG_SIZE = 640
N_MOSAICS = 5000

IMAGES_DIR = r"no_aug/Train"
COCO_JSON_PATH = r"no_aug/instances_Train.json"
OUTPUT_IMAGES_DIR = r"no_aug/images"
OUTPUT_COCO_JSON_PATH = r"no_aug/annotations_mosaic.json"

ROTATE_PROB = 0.5

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco


def build_index(coco):
    images = coco["images"]
    annotations = coco["annotations"]

    imgid_to_img = {img["id"]: img for img in images}
    imgid_to_anns = {}
    for ann in annotations:
        imgid_to_anns.setdefault(ann["image_id"], []).append(ann)

    return imgid_to_img, imgid_to_anns


def extract_image_classes(imgid_to_anns):
    imgid_to_cats = {}
    for img_id, anns in imgid_to_anns.items():
        cats = {ann["category_id"] for ann in anns if ann.get("iscrowd", 0) == 0}
        imgid_to_cats[img_id] = cats
    return imgid_to_cats


def collect_fruit_groups(coco):
    fruit_groups = {
        "apple": set(),
        "banana": set(),
        "orange": set(),
    }

    for cat in coco.get("categories", []):
        name = cat.get("name", "").lower()
        cid = cat["id"]

        if "apple" in name:
            fruit_groups["apple"].add(cid)
        if "banana" in name:
            fruit_groups["banana"].add(cid)
        if "orange" in name:
            fruit_groups["orange"].add(cid)

    return fruit_groups


def rotate_point_90_cw(x, y, w, h):
    x_new = h - y
    y_new = x
    return x_new, y_new


def rotate_segmentation_90_cw(segmentation, w, h):
    new_segs = []

    for poly in segmentation:
        new_poly = []
        for i in range(0, len(poly), 2):
            x, y = poly[i], poly[i + 1]
            x_n, y_n = rotate_point_90_cw(x, y, w, h)
            new_poly.extend([x_n, y_n])
        new_segs.append(new_poly)

    return new_segs


def rotate_bbox_90_cw(bbox, w, h):
    x, y, bw, bh = bbox
    pts = [
        (x, y),
        (x + bw, y),
        (x + bw, y + bh),
        (x, y + bh),
    ]

    rot_pts = [rotate_point_90_cw(px, py, w, h) for px, py in pts]
    xs = [p[0] for p in rot_pts]
    ys = [p[1] for p in rot_pts]

    x_new = min(xs)
    y_new = min(ys)
    bw_new = max(xs) - x_new
    bh_new = max(ys) - y_new

    return [x_new, y_new, bw_new, bh_new]


def rotate_image_and_anns_90_cw(img, anns):
    h, w = img.shape[:2]
    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new_w, new_h = h, w

    anns_rot = []
    for ann in anns:
        ann2 = deepcopy(ann)
        ann2["bbox"] = rotate_bbox_90_cw(ann["bbox"], w, h)

        if "segmentation" in ann and ann["segmentation"]:
            ann2["segmentation"] = rotate_segmentation_90_cw(ann["segmentation"], w, h)

        anns_rot.append(ann2)

    return img_rot, anns_rot, new_w, new_h


def transform_bbox_scale_translate(bbox, scale, dx, dy):
    x, y, w, h = bbox
    x1 = x * scale + dx
    y1 = y * scale + dy
    w1 = w * scale
    h1 = h * scale
    return [x1, y1, w1, h1]


def transform_segmentation_scale_translate(segmentation, scale, dx, dy):
    new_segs = []

    for poly in segmentation:
        new_poly = []
        for i in range(0, len(poly), 2):
            x = poly[i] * scale + dx
            y = poly[i + 1] * scale + dy
            new_poly.extend([x, y])
        new_segs.append(new_poly)

    return new_segs


def generate_layout(num_imgs, img_size):
    if num_imgs == 2:
        split = random.randint(int(img_size * 0.3), int(img_size * 0.7))

        if random.random() < 0.5:
            rects = [
                (0, 0, split, img_size),
                (split, 0, img_size - split, img_size),
            ]
        else:
            rects = [
                (0, 0, img_size, split),
                (0, split, img_size, img_size - split),
            ]

    elif num_imgs == 3:
        split_x = random.randint(int(img_size * 0.3), int(img_size * 0.7))
        split_y = random.randint(int(img_size * 0.3), int(img_size * 0.7))

        if random.random() < 0.5:
            rects = [
                (0, 0, split_x, img_size),
                (split_x, 0, img_size - split_x, split_y),
                (split_x, split_y, img_size - split_x, img_size - split_y),
            ]
        else:
            rects = [
                (0, 0, img_size - split_x, split_y),
                (0, split_y, img_size - split_x, img_size - split_y),
                (img_size - split_x, 0, split_x, img_size),
            ]

    else:
        split_x = random.randint(int(img_size * 0.3), int(img_size * 0.7))
        split_y = random.randint(int(img_size * 0.3), int(img_size * 0.7))

        rects = [
            (0, 0, split_x, split_y),
            (split_x, 0, img_size - split_x, split_y),
            (0, split_y, split_x, img_size - split_y),
            (split_x, split_y, img_size - split_x, img_size - split_y),
        ]

    return rects


def sample_images_for_mosaic(
    images_list,
    imgid_to_cats,
    banana_cat_ids,
    min_imgs=2,
    max_imgs=4,
    max_tries=20,
):
    valid_images = [img for img in images_list if imgid_to_cats.get(img["id"])]

    if len(valid_images) < min_imgs:
        return []

    for _ in range(max_tries):
        target_n = random.randint(min_imgs, max_imgs)
        random.shuffle(valid_images)

        chosen = []
        used_cats = set()
        banana_used = False

        for img in valid_images:
            img_id = img["id"]
            cats = imgid_to_cats.get(img_id, set())

            if not cats:
                continue

            if used_cats & cats:
                continue

            has_banana = bool(cats & banana_cat_ids)

            if banana_used and has_banana:
                continue

            chosen.append(img)
            used_cats |= cats

            if has_banana:
                banana_used = True

            if len(chosen) >= target_n:
                break

        if len(chosen) >= min_imgs:
            return chosen

    return []


def create_mosaic(
    images_list,
    imgid_to_anns,
    imgid_to_cats,
    banana_cat_ids,
    img_size,
    next_image_id,
    next_ann_id,
):
    chosen_imgs = sample_images_for_mosaic(
        images_list,
        imgid_to_cats,
        banana_cat_ids,
        min_imgs=2,
        max_imgs=4,
    )

    num_imgs = len(chosen_imgs)

    if num_imgs < 2:
        return None, None, [], next_image_id, next_ann_id

    mosaic = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    rects = generate_layout(num_imgs, img_size)
    all_new_anns = []

    for img_info, rect in zip(chosen_imgs, rects):
        qx, qy, qw, qh = rect
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(IMAGES_DIR, file_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        anns = imgid_to_anns.get(img_id, [])
        h0, w0 = img.shape[:2]

        if random.random() < ROTATE_PROB:
            img, anns, w_rot, h_rot = rotate_image_and_anns_90_cw(img, anns)
            w0, h0 = w_rot, h_rot

        sx = qw / float(w0)
        sy = qh / float(h0)
        scale = min(sx, sy)

        new_w = max(1, int(w0 * scale))
        new_h = max(1, int(h0 * scale))

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = qx + (qw - new_w) // 2
        pad_y = qy + (qh - new_h) // 2

        mosaic[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized

        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue

            bbox_scaled = transform_bbox_scale_translate(
                ann["bbox"],
                scale,
                pad_x,
                pad_y,
            )

            x, y, w, h = bbox_scaled

            if w < 1 or h < 1:
                continue

            new_ann = {
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": ann["category_id"],
                "bbox": bbox_scaled,
                "iscrowd": 0,
            }

            if "segmentation" in ann and ann["segmentation"]:
                seg_scaled = transform_segmentation_scale_translate(
                    ann["segmentation"],
                    scale,
                    pad_x,
                    pad_y,
                )
                new_ann["segmentation"] = seg_scaled

            orig_area = ann.get("area", w * h)
            area_scale = scale * scale
            new_ann["area"] = float(orig_area * area_scale)

            all_new_anns.append(new_ann)
            next_ann_id += 1

    new_image_info = {
        "id": next_image_id,
        "file_name": f"mosaic_{next_image_id}.jpg",
        "width": img_size,
        "height": img_size,
    }

    return mosaic, new_image_info, all_new_anns, next_image_id + 1, next_ann_id


def main():
    coco = load_coco(COCO_JSON_PATH)

    imgid_to_img, imgid_to_anns = build_index(coco)
    imgid_to_cats = extract_image_classes(imgid_to_anns)

    images_list = list(imgid_to_img.values())
    annotations = coco["annotations"]

    fruit_groups = collect_fruit_groups(coco)
    banana_cat_ids = fruit_groups.get("banana", set())

    max_img_id = max(img["id"] for img in images_list) if images_list else 0
    max_ann_id = max(ann["id"] for ann in annotations) if annotations else 0

    next_image_id = max_img_id + 1
    next_ann_id = max_ann_id + 1

    new_images = []
    new_annotations = []

    for _ in range(N_MOSAICS):
        mosaic_img, img_info, anns, next_image_id, next_ann_id = create_mosaic(
            images_list,
            imgid_to_anns,
            imgid_to_cats,
            banana_cat_ids,
            IMG_SIZE,
            next_image_id,
            next_ann_id,
        )

        if mosaic_img is None or len(anns) == 0:
            continue

        out_path = os.path.join(OUTPUT_IMAGES_DIR, img_info["file_name"])
        cv2.imwrite(out_path, mosaic_img)

        new_images.append(img_info)
        new_annotations.extend(anns)

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": new_images,
        "annotations": new_annotations,
    }

    with open(OUTPUT_COCO_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
