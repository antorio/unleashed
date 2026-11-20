"""Target skin texture (Swap box, unleashed.globals.target_texture, %, 0 = off).

inswapper renders the face at 128 px (AlphaFace 256): pores, freckles, spots,
sweat shine and the footage's own grain of the TARGET are gone, so the swapped
skin looks smooth and pasted-on. This lays a thin layer of the target's fine
detail back over the swapped skin:

    detail = Y(target crop) - GaussianBlur(Y(target crop), sigma)   (brightness only)
    out    = swapped + amount * skin_mask * detail                  (all 3 channels)

Only the high frequencies are taken (sigma ~1% of the crop), so the shape,
the colour and the lighting of the swapped face stay as they are. The target
crop is the very crop the swap was made from (same alignment), so the detail
lands where it was in the target. Eyes, brows, lips and nostrils are left out
of the mask -- their detail would draw the target's features over the
swapped ones -- and the mask is feathered.
"""
import cv2
import numpy as np

SIGMA_FRAC = 0.012        # blur radius as a fraction of the crop size (~1.5 px at 128, 3 px at 256)


def _poly(mask, pts, value=1.0):
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], value)


def skin_mask(pts, size):
    """Feathered skin mask (size x size, float32 0..1) from the 68 points in
    crop coordinates: the face up to the brows, minus eyes, brows, lips and
    nostrils."""
    m = np.zeros((size, size), np.float32)
    iod = float(np.linalg.norm(pts[36:42].mean(0) - pts[42:48].mean(0))) + 1e-6
    jaw = pts[0:17]
    if not np.all(np.isfinite(jaw)):
        # only the inner points (read from the swapped crop): the hull of
        # brows..mouth, widened, stands in for the jaw line
        inner = pts[17:68]
        c = inner.mean(0)
        hull = cv2.convexHull(np.round(c + (inner - c) * 1.25).astype(np.float32)).reshape(-1, 2)
        _poly(m, hull)
        jaw = None
    brows = pts[17:27]
    # the forehead: the brows raised by ~0.6 eye distance
    up = (pts[27] - pts[57])
    up = up / (np.linalg.norm(up) + 1e-6)
    forehead = brows[::-1] + up * 0.6 * iod
    if jaw is not None:
        _poly(m, np.concatenate([jaw, forehead]))
    else:
        _poly(m, np.concatenate([brows, forehead]))
    holes = np.zeros_like(m)
    for eye in (pts[36:42], pts[42:48]):
        c = eye.mean(0)
        _poly(holes, c + (eye - c) * 1.6)
    for brow in (pts[17:22], pts[22:27]):
        cv2.polylines(holes, [np.round(brow).astype(np.int32)], False, 1.0, max(2, int(0.18 * iod)))
    lips = pts[48:60]
    c = lips.mean(0)
    _poly(holes, c + (lips - c) * 1.25)
    nostrils = pts[31:36]
    c = nostrils.mean(0)
    cv2.ellipse(holes, (int(round(c[0])), int(round(c[1] - 0.05 * iod))),
                (max(2, int(0.42 * iod)), max(2, int(0.16 * iod))), 0, 0, 360, 1.0, -1)
    m = np.clip(m - holes, 0.0, 1.0)
    k = max(1.0, 0.08 * iod)
    m = cv2.GaussianBlur(m, (0, 0), k)
    return m


def apply(swapped, target_crop, pts, amount):
    """swapped / target_crop: the swapped face crop and the target crop it was
    made from (same size, BGR uint8). pts: 68 x 2 in crop coordinates.
    amount: 0..100 (%). Returns uint8."""
    a = float(amount) / 100.0
    if a <= 0.0 or pts is None or swapped.shape != target_crop.shape:
        return swapped
    size = swapped.shape[0]
    y = cv2.cvtColor(target_crop, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    sigma = max(0.8, SIGMA_FRAC * size)
    detail = y - cv2.GaussianBlur(y, (0, 0), sigma)
    m = skin_mask(np.asarray(pts, np.float64), size)
    out = swapped.astype(np.float32) + (a * m * detail)[..., None]
    return np.clip(out, 0, 255).round().astype(np.uint8)
