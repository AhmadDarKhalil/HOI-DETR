"""
vis_offline.py
--------------
Re-render HOI visualisations from an exported predictions JSON, without
running the model. Works with both export types:

    * image export  (predictions.json written by demo.py)
    * video export  (<name>.json written by demo_video.py)

It produces the same annotated images / mp4s as the live demos, reading
boxes and interactions straight from the JSON. Only the source pixels
(the original images / video) are needed -- no checkpoint, no GPU.

Edit the variables at the top for your paths, then run:
    python demo/vis_offline.py
"""

import json
import os

import cv2
import mmcv
from tqdm import tqdm

from helpers import compute_style, draw_ui
from predictions_io import load_detections, load_pairs


# ══════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════
# Path to a predictions JSON exported by demo.py or demo_video.py.
PREDICTIONS_JSON = 'demo/results/example_videos/food_tour.json'

# Output: None -> demo/results/offline/<json_stem>/  (recommended)
#         str  -> use that exact directory
OUTPUT_DIR = None

# Source override. The JSON records the original image dir / video path it
# was exported from; set this if those files have since moved.
#   image JSON -> a directory of source images
#   video JSON -> a single source video file
# None -> use the path stored in the JSON.
SOURCE_OVERRIDE = None

# Visualisation mode (see demo.py for details).
VERBOSE_LABELS = True

# Output codec / container for video re-rendering.
FOURCC = 'mp4v'


# ══════════════════════════════════════════════════════════════
# Image re-rendering
# ══════════════════════════════════════════════════════════════
def render_images(meta, out_dir):
    src_dir = SOURCE_OVERRIDE or meta.get('input_dir', '')
    print(f"[INFO] {len(meta['images'])} image(s) from {src_dir}")

    for rec in tqdm(meta['images']):
        img_path = os.path.join(src_dir, rec['file_name'])
        img = mmcv.imread(img_path)
        if img is None:
            print(f"[WARN] cannot read {img_path}, skipping")
            continue

        dets = load_detections(rec['detections'])
        hf   = load_pairs(rec['hf'], dets)
        fs   = load_pairs(rec['fs'], dets)

        vis = img.copy()
        draw_ui(vis, dets, hf, fs, compute_style(vis.shape),
                verbose_labels=VERBOSE_LABELS)
        mmcv.imwrite(vis, os.path.join(out_dir, rec['file_name']))


# ══════════════════════════════════════════════════════════════
# Video re-rendering (mirrors demo_video.py's writer loop)
# ══════════════════════════════════════════════════════════════
def render_video(meta, out_dir):
    src = SOURCE_OVERRIDE or meta['video_path']
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] cannot open source video {src}")
        return

    fps    = meta.get('fps') or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = meta.get('width')  or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = meta.get('height') or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem     = os.path.splitext(os.path.basename(src))[0]
    out_path = os.path.join(out_dir, f'{stem}.mp4')
    fourcc   = cv2.VideoWriter_fourcc(*FOURCC)
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"[ERROR] cannot open writer for {out_path}")
        return

    print(f"[INFO] {len(meta['frames'])} frame(s) from {src}")

    # One record per source frame. Only `processed` frames carry fresh
    # detections; skipped frames repeat the last rendered frame, exactly
    # as demo_video.py wrote them.
    last_vis = None
    try:
        for rec in tqdm(meta['frames'], desc=stem, leave=False):
            ok, frame = cap.read()
            if not ok:
                break

            if rec['processed']:
                dets = load_detections(rec['detections'])
                hf   = load_pairs(rec['hf'], dets)
                fs   = load_pairs(rec['fs'], dets)
                vis  = frame.copy()
                draw_ui(vis, dets, hf, fs, compute_style(vis.shape),
                        verbose_labels=VERBOSE_LABELS)
                last_vis = vis
            else:
                vis = last_vis if last_vis is not None else frame

            writer.write(vis)
    finally:
        cap.release()
        writer.release()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    with open(PREDICTIONS_JSON) as f:
        meta = json.load(f)

    # Dispatch on the explicit type, falling back to the payload shape.
    kind = meta.get('type')
    if kind is None:
        kind = 'video' if 'frames' in meta else 'image'

    stem    = os.path.splitext(os.path.basename(PREDICTIONS_JSON))[0]
    out_dir = OUTPUT_DIR or os.path.join('demo', 'results', 'offline', stem)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] {kind} export -> {out_dir}")

    if kind == 'video':
        render_video(meta, out_dir)
    else:
        render_images(meta, out_dir)

    print(f"[INFO] Done. Results saved to {out_dir}")


if __name__ == '__main__':
    main()
