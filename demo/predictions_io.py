"""
predictions_io.py
-----------------
Shared (de)serialisation for HOI prediction JSON, used by:
    demo.py          (image export)
    demo_video.py    (video export)
    vis_offline.py   (offline re-rendering from an exported JSON)

A detection is exported with only these fields:
    box        xyxy in original-image pixel coords
    score      detection confidence (0-1)
    class_id   0=hand, 1=firstobject, 2=secondobject
    class_name string form of class_id

Interactions (hf, fs) reference their two endpoints by index into the
frame/image `detections` list, plus the interaction prob -- no detection
data is duplicated. Render-only keys that draw_ui attaches in place
(draw_box, label_prefer, coincident_inner/outer) and internal fields
(center, query_idx) are intentionally dropped.
"""

import numpy as np


# ── export ────────────────────────────────────────────────────
def det_to_jsonable(d):
    """Whitelist the meaningful fields of a detection dict for JSON."""
    return {
        'box':        [float(x) for x in d['box']],   # xyxy, original-image px
        'score':      float(d['score']),
        'class_id':   int(d['class_id']),
        'class_name': d['class_name'],
    }


def detections_record(dets, hf, fs):
    """
    Serialise one frame/image's detections and interactions.

    Returns (detections_json, hf_json, fs_json) where hf/fs reference the
    two endpoints by their index in detections_json.
    """
    det_index = {id(d): i for i, d in enumerate(dets)}

    def _pair(pair):
        a, b, prob = pair
        return {'a': det_index[id(a)], 'b': det_index[id(b)],
                'prob': round(float(prob), 4)}

    return (
        [det_to_jsonable(d) for d in dets],
        [_pair(p) for p in hf],
        [_pair(p) for p in fs],
    )


# ── import (for offline re-rendering) ─────────────────────────
def load_detections(det_json):
    """
    Rebuild detection dicts from JSON so they are drop-in compatible with
    draw_ui: box is restored to a numpy array and center (the box midpoint,
    which draw_ui's links need) is recomputed exactly as run_inference did.
    """
    dets = []
    for d in det_json:
        box = np.array(d['box'], dtype=np.float32)
        cx  = int((box[0] + box[2]) / 2)
        cy  = int((box[1] + box[3]) / 2)
        dets.append(dict(
            box        = box,
            score      = float(d['score']),
            class_id   = int(d['class_id']),
            class_name = d['class_name'],
            center     = (cx, cy),
        ))
    return dets


def load_pairs(pair_json, dets):
    """Resolve index-referenced pairs back into (det_a, det_b, prob) tuples."""
    return [(dets[p['a']], dets[p['b']], float(p['prob'])) for p in pair_json]
