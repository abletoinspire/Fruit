# python inference_video_A.py --weights ./runs_rotten_only/weights/best.pth --input ./input.mp4 --output ./output_annotated.mp4 --fruit-thr 0.55 --min-area 700 --split auto --core-erosion 5 --core-prob-thr 0.70 --core-min-px 200
import os
import time
import argparse

import cv2
import torch

from inference_A import UNetRottenOnly, infer_one, get_device_type, AMP_AVAILABLE


def select_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(weights_path: str, device: str, channels_last: bool):
    ckpt = torch.load(weights_path, map_location="cpu")
    quality_names = ckpt.get("quality_names", ["fresh", "rotten"])
    if not isinstance(quality_names, (list, tuple)) or len(quality_names) < 2:
        quality_names = ["fresh", "rotten"]

    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    img_size = int(cfg.get("img_size", 512))
    base_c = int(cfg.get("base_c", 48))

    model = UNetRottenOnly(base_c=base_c, num_qualities=len(quality_names), head_drop=0.10).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    if get_device_type(device) == "cuda" and channels_last:
        model = model.to(memory_format=torch.channels_last)

    return model, list(quality_names), img_size


def open_writer(path: str, fps: float, width: int, height: int, codec: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть VideoWriter для файла: {path}")
    return writer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fruit-thr", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=700)
    ap.add_argument("--split", choices=["auto", "cc", "watershed"], default="auto")
    ap.add_argument("--ws-fg-frac", type=float, default=0.45)
    ap.add_argument("--ws-bg-dilate", type=int, default=2)
    ap.add_argument("--cls-thr", type=float, default=0.35)
    ap.add_argument("--only-rotten", action="store_true")
    ap.add_argument("--rotten-score-thr", type=float, default=0.50)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--core-erosion", type=int, default=5)
    ap.add_argument("--core-prob-thr", type=float, default=0.70)
    ap.add_argument("--core-min-px", type=int, default=200)
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--codec", default="mp4v")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--frame-step", type=int, default=1)
    args = ap.parse_args()

    device = select_device(args.device)
    device_type = get_device_type(device)

    if device_type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model, quality_names, img_size = load_model(args.weights, device, args.channels_last)
    use_amp = bool(args.use_amp and AMP_AVAILABLE and device_type == "cuda")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = None
    frame_id = 0
    written = 0
    last_out = None
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        if args.max_frames > 0 and frame_id > args.max_frames:
            break

        process_this = ((frame_id - 1) % max(1, args.frame_step) == 0)
        if process_this or last_out is None:
            out_frame, _, _, _, _ = infer_one(
                model=model,
                img_bgr=frame,
                img_size=img_size,
                quality_names=quality_names,
                device=device,
                use_amp=use_amp,
                channels_last=args.channels_last,
                fruit_thr=args.fruit_thr,
                min_area=args.min_area,
                split=args.split,
                ws_fg_frac=args.ws_fg_frac,
                ws_bg_dilate=args.ws_bg_dilate,
                cls_thr=args.cls_thr,
                only_rotten=args.only_rotten,
                rotten_score_thr=args.rotten_score_thr,
                use_open=(not args.no_open),
                core_erosion=args.core_erosion,
                core_prob_thr=args.core_prob_thr,
                core_min_px=args.core_min_px,
                panel=args.panel,
            )
            last_out = out_frame
        else:
            last_out = frame

        if writer is None:
            h, w = last_out.shape[:2]
            writer = open_writer(args.output, fps, w, h, args.codec)

        writer.write(last_out)
        written += 1

        if written % 50 == 0:
            elapsed = max(1e-6, time.time() - t0)
            suffix = f"/{total}" if total > 0 else ""
            print(f"Кадры: {written}{suffix}, скорость: {written / elapsed:.2f} кадр/с")

    cap.release()
    if writer is not None:
        writer.release()

    print(f"Готово. Сохранено: {args.output}")


if __name__ == "__main__":
    main()