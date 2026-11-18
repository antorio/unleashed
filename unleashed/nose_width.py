"""Nose width (Swap box, unleashed.globals.nose_width, % , 0 = off).

Measured 18 Nov (30 pairs from insightface's t1.jpg, factory settings): the
swapped nose is on average 1.8% wider than the target's in the same pose
(alar points 31-35 of the 68-point scheme / outer eye corners), the mouth
2.3% -- about 1 px per side on a face 300 px tall. This narrows (or widens)
the nose of the swapped face crop horizontally, right after the swap and
before the expression restorer / mask / enhancer:

    output x = cx + (x - cx) * (1 + s * w),   s = nose_width / 100

cx = middle of the nostril wings, w = 1 over the nose (to 1.2x its half-width:
the wings reach a little past points 31 / 35), fading smoothly to 0 at twice
the half-width sideways, at the root of
the nose above and at the top of the upper lip below, so cheeks and lips
barely move and nothing outside the region changes at all. The points are
the target's 68 landmarks (the ones the alignment used, smoothed in videos),
mapped into the crop with the alignment matrix -- no extra model run. Only
when the face has none (68-point alignment off) are they read from the
swapped crop itself.
"""
import cv2
import numpy as np

LIMIT = 15.0          # % either way (the slider's range)


def crop_points(target_face, M):
    """The target's 68 landmarks in crop coordinates, or None."""
    lm = None
    try:
        lm = target_face.get('landmark_3d_68')
    except Exception:
        lm = getattr(target_face, 'landmark_3d_68', None)
    if lm is None:
        return None
    lm = np.asarray(lm, np.float64)
    if lm.ndim != 2 or lm.shape[0] != 68:
        return None
    xy = lm[:, :2]
    M = np.asarray(M, np.float64)
    return xy @ M[:, :2].T + M[:, 2]


def warp(img, pts, amount):
    """img: the swapped face crop (H x W x 3). pts: 68 x 2 points in crop
    coordinates. amount: % (negative = narrower). Returns a new image, or img
    itself when there is nothing to do."""
    s = float(np.clip(amount, -LIMIT, LIMIT)) / 100.0
    if abs(s) < 1e-6 or pts is None:
        return img
    pts = np.asarray(pts, np.float64)
    h, w = img.shape[:2]
    left, right = pts[31], pts[35]
    cx = 0.5 * (left[0] + right[0])
    half = 0.5 * abs(right[0] - left[0])
    ny = float(np.mean(pts[31:36, 1]))
    up = ny - float(pts[27, 1])                 # to the root of the nose
    down = float(pts[51, 1]) - ny               # to the top of the upper lip
    if not (half > 1.0 and up > 1.0):
        return img
    down = max(down, 0.2 * up)
    rx = 2.0 * half
    x0, x1 = int(np.floor(cx - rx)), int(np.ceil(cx + rx)) + 1
    y0, y1 = int(np.floor(ny - up)), int(np.ceil(ny + down)) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return img
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    dx = (X - cx) / rx
    dy = np.where(Y < ny, (Y - ny) / up, (Y - ny) / down)
    r = np.sqrt(dx * dx + dy * dy)
    t = np.clip((r - 0.6) / 0.4, 0.0, 1.0)
    wgt = 1.0 - t * t * (3.0 - 2.0 * t)          # 1 inside, smoothstep to 0 at r = 1
    # inverse map: the output pixel at x shows the input at cx + (x - cx) / (1 + s w)
    map_x = (cx + (X - cx) / (1.0 + s * wgt)).astype(np.float32)
    map_y = Y.astype(np.float32)
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REFLECT)
    return out
