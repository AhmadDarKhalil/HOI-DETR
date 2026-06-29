"""
demo_video.py
-------------
Co-DETR hand-object interaction demo for videos.

Takes a folder of mp4 videos and writes annotated mp4 videos to an output
folder, processing frame-by-frame with the same HOI pipeline as demo.py.

Also writes a per-video JSON file capturing detections and interactions
for every frame, alongside the mp4.

Edit the variables at the top for your paths, then run:
    python demo/demo_video.py
"""

import glob
import json
import os

import cv2
import mmcv
from tqdm import tqdm

from mmdet.apis               import init_detector
from mmdet.datasets.pipelines import Compose

from projects import *  # noqa: F401,F403  (registers Co-DETR custom modules)

from configs import CLASS_NAMES
from helpers import (
    find_interaction_branch,
    run_inference,
    call_interaction,
    compute_style,
    draw_ui,
)
from predictions_io import detections_record


# ══════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════
MODEL_CONFIG = 'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py'
CHECKPOINT   = 'checkpoints/epoch_5.pth'
DEVICE       = 'cuda:0'

# Input: a directory of mp4 videos (any folder name; final path segment is
# reused in the default output directory).
INPUT_DIR    = 'demo/example_videos'

# Output: None  -> demo/results/<basename(INPUT_DIR)>/  (recommended)
#         str   -> use that exact directory
OUTPUT_DIR   = None

# Detection thresholds
SCORE_THR    = 0.3
NMS_IOU      = 0.5

# Visualisation mode
# VERBOSE_LABELS = False -> smart hiding: labels/badges on tiny boxes and
#                           short links are suppressed (cleaner output).
# VERBOSE_LABELS = True  -> show every label and every probability badge,
#                           regardless of size (use for debugging or when
#                           you want the raw, complete view).
VERBOSE_LABELS = True

# Frame sampling: process every Nth frame (1 = every frame). Skipped
# frames are still written to the output (without overlays) so the
# output preserves the source duration. Set to 1 for full quality.
FRAME_STRIDE = 1

# Output codec / container. mp4v is widely compatible; if you have
# ffmpeg-built OpenCV you can switch to 'avc1' (H.264) for smaller files.
FOURCC = 'mp4v'

# Export predictions: when True, write a per-video <name>.json capturing
# detections and interactions for every frame (see predictions_io.py for
# the schema). Set False to write only the annotated .mp4. The exported
# JSON can be re-rendered later with vis_offline.py.
EXPORT_JSON = True


# ══════════════════════════════════════════════════════════════
# Per-frame HOI processing (mirrors the loop body in demo.py)
# ══════════════════════════════════════════════════════════════
def process_frame(frame, model, test_pipeline, interaction_branch, tmp_path):
    """
    Run detection + interaction on a single BGR frame.
    Returns (annotated_frame, dets, hf_inters, fs_inters) so the caller
    can both write the visualisation and log results to JSON.
    """
    # mmdet's pipeline expects a file path, so write the frame to a temp file.
    mmcv.imwrite(frame, tmp_path)

    try:
        dets, embeds = run_inference(
            model, test_pipeline, tmp_path,
            device      = DEVICE,
            class_names = CLASS_NAMES,
            score_thr   = SCORE_THR,
            nms_iou     = NMS_IOU,
        )
    except Exception as e:
        print(f"[WARN] inference failed on frame: {e}")
        return frame, [], [], []

    if not dets:
        return frame, [], [], []

    # Predict interactions: all H->F and F->S pairs
    hands   = [d for d in dets if d['class_id'] == 0]
    firsts  = [d for d in dets if d['class_id'] == 1]
    seconds = [d for d in dets if d['class_id'] == 2]

    hf_inters, fs_inters = [], []
    for h in hands:
        for f in firsts:
            interacts, prob = call_interaction(
                interaction_branch,
                embeds[h['query_idx']], embeds[f['query_idx']],
            )
            if interacts:
                if prob > 0.6:
                    hf_inters.append((h, f, prob))
    for f in firsts:
        for so in seconds:
            interacts, prob = call_interaction(
                interaction_branch,
                embeds[f['query_idx']], embeds[so['query_idx']],
            )
            if interacts:
                if prob > 0.92:
                    fs_inters.append((f, so, prob))

    vis   = frame.copy()
    style = compute_style(vis.shape)
    draw_ui(vis, dets, hf_inters, fs_inters, style,
            verbose_labels=VERBOSE_LABELS)
    return vis, dets, hf_inters, fs_inters


# ══════════════════════════════════════════════════════════════
# Per-video processing
# ══════════════════════════════════════════════════════════════
def process_video(video_path, out_path, json_path, model, test_pipeline,
                  interaction_branch, tmp_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] cannot open {video_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"[ERROR] cannot open writer for {out_path}")
        return False

    # JSON accumulator: video-level meta + a list of per-frame records.
    # Only built when exporting.
    meta = None
    if EXPORT_JSON:
        meta = {
            'type':         'video',
            'video_path':   os.path.abspath(video_path),
            'fps':          float(fps),
            'width':        width,
            'height':       height,
            'num_frames':   total,
            'score_thr':    SCORE_THR,
            'nms_iou':      NMS_IOU,
            'frame_stride': FRAME_STRIDE,
            'class_names':  list(CLASS_NAMES),
            'frames':       [],
        }

    last_vis = None
    last_dets, last_hf, last_fs = [], [], []
    pbar = tqdm(total=total, desc=os.path.basename(video_path), leave=False)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % FRAME_STRIDE == 0:
                vis, dets, hf, fs = process_frame(
                    frame, model, test_pipeline,
                    interaction_branch, tmp_path,
                )
                last_vis = vis
                last_dets, last_hf, last_fs = dets, hf, fs
                processed = True
            else:
                # Reuse last annotated frame to keep overlays visually
                # stable on skipped frames; if none yet, write raw.
                vis = last_vis if last_vis is not None else frame
                dets, hf, fs = last_dets, last_hf, last_fs
                processed = False

            writer.write(vis)

            # Log this frame's results (mark whether inferred or copied
            # from the previous processed frame so consumers can tell).
            if EXPORT_JSON:
                det_j, hf_j, fs_j = detections_record(dets, hf, fs)
                meta['frames'].append({
                    'frame_idx':  idx,
                    'processed':  processed,
                    'detections': det_j,
                    'hf':         hf_j,
                    'fs':         fs_j,
                })

            idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()

    # Write JSON alongside the mp4.
    if EXPORT_JSON:
        with open(json_path, 'w') as f:
            json.dump(meta, f)

    return True


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    # Resolve output directory
    out_dir = OUTPUT_DIR or os.path.join(
        'demo', 'results', os.path.basename(os.path.normpath(INPUT_DIR))
    )
    os.makedirs(out_dir, exist_ok=True)

    # Collect input videos (preserve filenames on output, force .mp4)
    exts = ('*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.mkv')
    video_list = sorted(
        f for ext in exts for f in glob.glob(os.path.join(INPUT_DIR, ext))
    )
    if not video_list:
        print(f"[ERROR] No videos found in {INPUT_DIR}")
        return
    print(f"[INFO] {len(video_list)} video(s) from {INPUT_DIR}")
    print(f"[INFO] Saving to {out_dir}")

    # Build model and pipeline
    model = init_detector(MODEL_CONFIG, CHECKPOINT, device=DEVICE)
    model.eval()
    test_pipeline      = Compose(model.cfg.data.test.pipeline)
    interaction_branch = find_interaction_branch(model.query_head)
    print(f"[INFO] Interaction MLP input dim: "
          f"{interaction_branch.mlp[0].in_features}")

    # Track temp files so we can delete them all at the end. Each video
    # gets its own scratch image named after the video stem so multiple
    # instances of this script can run in parallel without clashing.
    tmp_paths = []

    # Main loop
    for video_path in tqdm(video_list, desc='videos'):
        stem      = os.path.splitext(os.path.basename(video_path))[0]
        out_path  = os.path.join(out_dir, f'{stem}.mp4')
        json_path = os.path.join(out_dir, f'{stem}.json')
        tmp_path  = os.path.join(out_dir, f'._frame_tmp_{stem}.jpg')
        tmp_paths.append(tmp_path)

        try:
            process_video(video_path, out_path, json_path,
                          model, test_pipeline,
                          interaction_branch, tmp_path)
        except Exception as e:
            print(f"[ERROR] {video_path}: {e}")
            continue

    # Clean up scratch files
    for tmp_path in tmp_paths:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    print(f"[INFO] Done. Results saved to {out_dir}")


if __name__ == '__main__':
    main()