"""Misalignment score: how far the outline of the observed anatomy is from the outline rendered at the current
registration.

Object agnostic: it compares two binary masks of one view, the observed one (SAM) and the one rendered at the current
registration, whatever the object (face, mandible). For each outline pixel of one mask, the distance to the nearest
outline pixel of the other (distance transform), both ways; distances in mm on the object plane (px x depth / focal,
mm_per_px below), truncated at TRUNC_MM so that spurious outline pieces do not weigh too much; the score of the view is
the mean of the two directed means. Outline pixels near the image border or on known occluders (ignore mask) are left
out. Score of several views: mean of the view scores.

Threshold tau: 95th percentile of the score on pre-impact images (threshold); registration lost when the score is
above tau. detection() gives AUC, sensitivity and specificity against the true label (mean TRE without correction above
2 mm).
"""
import cv2
import numpy as np

TRUNC_MM = 3.0     # distances above this count as this
BORDER_PX = 5      # outline pixels this close to the image border are left out
CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def outline(mask):
    """Outline of a binary mask: pixels of the mask with a 4-neighbour outside it (bool image)."""
    m = np.asarray(mask) > 0
    return m & ~cv2.erode(m.astype(np.uint8), CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def mm_per_px(K, T, vertices):
    """Scale of the object plane: median depth of the vertices (camera from model T, mm) over the mean focal."""
    z = np.asarray(vertices) @ T[2, :3] + T[2, 3]
    return float(np.median(z) / ((K[0, 0] + K[1, 1]) / 2))


def kept(shape, border_px=BORDER_PX, ignore=None):
    """Pixels where outline points count: away from the image border and off the ignore mask (bool image)."""
    h, w = shape
    keep = np.zeros((h, w), bool)
    keep[border_px:h - border_px, border_px:w - border_px] = True
    if ignore is not None:
        keep &= ~(np.asarray(ignore) > 0)
    return keep


def view_score(observed, rendered, scale, ignore=None, border_px=BORDER_PX, trunc_mm=TRUNC_MM):
    """Score of one view (mm): observed and rendered binary masks of the same size, scale in mm per px.

    Returns score_mm, the two directed means (obs_to_ren_mm: observed outline to rendered outline, ren_to_obs_mm) and
    the number of outline pixels used. An empty outline gives trunc_mm.
    """
    masks = [m if m.dtype == np.uint8 else (m > 0).astype(np.uint8) for m in (np.asarray(observed), np.asarray(rendered))]
    boxes = [b for b in map(cv2.boundingRect, masks) if b[2] > 0]
    if not boxes:
        return {"score_mm": trunc_mm, "obs_to_ren_mm": trunc_mm, "ren_to_obs_mm": trunc_mm, "n_obs": 0, "n_ren": 0}
    h, w = masks[0].shape
    pad = int(np.ceil(trunc_mm / scale)) + 2     # crop to both masks: distances above trunc_mm are cut anyway
    x0, y0 = max(min(b[0] for b in boxes) - pad, 0), max(min(b[1] for b in boxes) - pad, 0)
    x1, y1 = min(max(b[0] + b[2] for b in boxes) + pad, w), min(max(b[1] + b[3] for b in boxes) + pad, h)
    a, b = (outline(m[y0:y1, x0:x1]) for m in masks)
    yy, xx = np.arange(y0, y1), np.arange(x0, x1)
    keep = ((yy >= border_px) & (yy < h - border_px))[:, None] & ((xx >= border_px) & (xx < w - border_px))[None, :]
    if ignore is not None:
        keep &= ~(np.asarray(ignore)[y0:y1, x0:x1] > 0)
    out = {}
    for name, src, dst in (("obs_to_ren", a, b), ("ren_to_obs", b, a)):
        pts = src & keep
        n = int(pts.sum())
        if not n or not dst.any():
            out[name + "_mm"], out["n_" + name[:3]] = trunc_mm, n
            continue
        d = cv2.distanceTransform(np.where(dst, 0, 255).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        out[name + "_mm"] = float(np.minimum(d[pts] * scale, trunc_mm).mean())
        out["n_" + name[:3]] = n
    out["score_mm"] = (out["obs_to_ren_mm"] + out["ren_to_obs_mm"]) / 2
    return out


def score(view_scores):
    """Score of several views: mean of the view scores (dicts from view_score or numbers)."""
    return float(np.mean([v["score_mm"] if isinstance(v, dict) else v for v in view_scores]))


def threshold(pre_impact_scores, q=95.0):
    """tau: q-th percentile of the score on pre-impact images."""
    return float(np.percentile(pre_impact_scores, q))


def roc_auc(scores, labels):
    """Area under the ROC curve (probability that a lost registration scores above a valid one, ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def detection(scores, labels, tau):
    """Detection at tau: AUC, sensitivity and specificity (label True = registration lost)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    flag = s > tau
    return {"auc": roc_auc(s, y), "sensitivity": float(flag[y].mean()) if y.any() else float("nan"),
            "specificity": float((~flag[~y]).mean()) if (~y).any() else float("nan"), "n_lost": int(y.sum()),
            "n_valid": int((~y).sum())}
