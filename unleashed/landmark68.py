"""High-accuracy 68-point facial landmarker (FaceFusion's 2dfan4, ONNX, MIT).

Stage-1 upgrade over buffalo_l's landmark_3d_68: when enabled (toggle
`use_hi_landmarker`) and the model file is present, the 2dfan4 68 landmarks
replace `face.landmark_3d_68` for every detected face. Those landmarks drive:
  - landmark-based swap alignment (when use_landmark_alignment is on), and
  - nothing else (the swap alignment is its only consumer).
The 5-point arcface keypoints (kps) used for the core inswapper warp are NOT
touched, so face identity is unaffected. buffalo_l's landmark_2d_106 (mouth
mask / forehead) is also left as-is.

The preprocessing/postprocessing replicates FaceFusion's detect_with_2dfan4
exactly (face_angle assumed 0; our pipeline already de-rotates faces).

Model: 2dfan4.onnx from facefusion-assets. Put it in ./models/2dfan4.onnx or set
unleashed.globals.hi_landmarker_model_path. If missing/unloadable it silently falls
back to buffalo_l landmarks (a one-time warning is printed).
"""

import os
import threading

import cv2
import numpy as np

import unleashed.globals

_LM68 = None
_LM68_FAILED = False          # remember a missing/broken model so we don't retry per frame
_LM68_LOCK = threading.Lock()


def _resolve_model_path():
    p = getattr(unleashed.globals, 'hi_landmarker_model_path', '') or ''
    if p and os.path.isfile(p):
        return p
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join('models', '2dfan4.onnx'),                       # cwd/models (repo root)
        os.path.join(here, '..', 'models', '2dfan4.onnx'),           # <repo>/models
        os.path.join(here, '..', 'models', 'landmark', '2dfan4.onnx'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def get_landmarker():
    """Lazily load the 2dfan4 session (thread-safe singleton). None on failure."""
    global _LM68, _LM68_FAILED
    if _LM68 is not None or _LM68_FAILED:
        return _LM68
    with _LM68_LOCK:
        if _LM68 is not None or _LM68_FAILED:
            return _LM68
        path = _resolve_model_path()
        if path is None:
            print("[hi-landmarker] 2dfan4.onnx not found -> using buffalo_l 68pts. "
                  "Download it to ./models/2dfan4.onnx (see notes) to enable.")
            _LM68_FAILED = True
            return None
        try:
            import onnxruntime
            try:
                from unleashed.utilities import tuned_execution_providers
                providers = tuned_execution_providers()
            except Exception:
                providers = getattr(unleashed.globals, 'execution_providers', None) \
                    or ['CPUExecutionProvider']
            sess = onnxruntime.InferenceSession(path, providers=providers)
            _LM68 = _Fan2d4(sess)
            print(f"[load] 2dfan4 hi-accuracy landmarker loaded: {path}")
        except Exception as e:
            print(f"[hi-landmarker] failed to load 2dfan4 ({e}) -> using buffalo_l 68pts.")
            _LM68_FAILED = True
            _LM68 = None
        return _LM68


class _Fan2d4:
    """Runs 2dfan4 on a face bbox, returns 68 xy landmarks in frame space + score.

    Mirrors FaceFusion detect_with_2dfan4 (no rotation branch):
        scale       = 195 / max(bbox_w, bbox_h)
        translation = (256 - (bbox_min+bbox_max)*scale) * 0.5
        crop        = warpAffine(frame, [[s,0,tx],[0,s,ty]], (256,256))
        blob        = crop -> CHW float32 / 255
        lm, heat    = model(blob)          # lm in 0..64 grid space
        lm          = lm/64*256 -> invertAffine -> frame space
    """
    MODEL_SIZE = 256

    def __init__(self, session):
        self.session = session
        try:
            self.input_name = session.get_inputs()[0].name
        except Exception:
            self.input_name = 'input'

    def detect(self, frame_bgr, bbox):
        b = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
        wh = float(np.maximum(np.max(b[2:] - b[:2]), 1.0))
        scale = 195.0 / wh
        tx, ty = (self.MODEL_SIZE - (b[2:] + b[:2]) * scale) * 0.5
        affine = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)

        crop = cv2.warpAffine(frame_bgr, affine, (self.MODEL_SIZE, self.MODEL_SIZE))
        crop = self._optimize_contrast(crop)
        blob = crop.transpose(2, 0, 1).astype(np.float32) / 255.0

        out = self.session.run(None, {self.input_name: [blob]})
        lm = np.asarray(out[0])[:, :, :2][0] / 64.0 * self.MODEL_SIZE      # [68,2] crop space
        lm = cv2.transform(lm.reshape(1, -1, 2),
                           cv2.invertAffineTransform(affine)).reshape(-1, 2)

        score = 1.0
        if len(out) > 1 and out[1] is not None:
            heat = np.asarray(out[1])
            if heat.ndim == 4:
                s = float(np.amax(heat, axis=(2, 3)).mean())
                score = float(np.interp(s, [0.0, 0.9], [0.0, 1.0]))
        return lm.astype(np.float32), score

    @staticmethod
    def _optimize_contrast(crop):
        # verbatim FaceFusion behaviour: round-trip is a no-op for normal frames,
        # CLAHE only kicks in on very dark crops.
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
        if np.mean(lab[:, :, 0]) < 30:
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


def refine_faces_landmark68(frame, faces):
    """If enabled, overwrite each face's landmark_3d_68 with 2dfan4's 68 points.

    Runs BEFORE temporal smoothing and BEFORE the swap, so the alignment
    consumes the higher-accuracy landmarks. z is preserved from buffalo_l's 3d68
    (2dfan4 is 2D). Any failure falls back to the existing landmarks silently.
    """
    if not getattr(unleashed.globals, 'use_hi_landmarker', False):
        return faces
    # The 68 points are consumed by the landmark alignment and nothing else, so
    # running 2dfan4 while alignment uses the detector kps would be pure wasted
    # inference (its result would be overwritten-then-ignored).
    if not getattr(unleashed.globals, 'use_landmark_alignment', True):
        return faces
    if not faces:
        return faces
    lm = get_landmarker()
    if lm is None:
        return faces

    prof = getattr(unleashed.globals, 'profile_timings', False)
    dbg = getattr(unleashed.globals, 'hi_landmarker_debug', False)
    t0 = None
    if prof:
        import time as _t
        t0 = _t.perf_counter()

    for f in faces:
        try:
            bbox = None
            try:
                bbox = f['bbox']
            except Exception:
                bbox = getattr(f, 'bbox', None)
            if bbox is None:
                continue
            pts, score = lm.detect(frame, bbox)
            if pts is None or pts.shape[0] != 68:
                continue
            # z is filled with zeros: 2dfan4 is a 2D landmarker, and nothing that
            # reads landmark_3d_68 uses the z column (landmark_68_to_5 takes
            # [:, :2] and the stabilizer leaves z untouched). buffalo_l's 1k3d68
            # runs alongside 2dfan4 only for face shape (kept below).
            z = np.zeros((68, 1), dtype=np.float32)
            # face shape from the source compares 3D shapes: it asks for
            # buffalo's 68 points (with depth) and reads this copy of them
            prev = f.get('landmark_3d_68') if hasattr(f, 'get') else None
            if prev is not None:
                f['landmark_3d_68_buffalo'] = prev
            f['landmark_3d_68'] = np.concatenate([pts.astype(np.float32), z], axis=1)
            if dbg:
                print(f"[hi-landmarker] refined 68pts (score={score:.2f})")
        except Exception as e:
            if dbg:
                print(f"[hi-landmarker] per-face refine failed: {e}")

    if prof and t0 is not None:
        import time as _t
        print(f"[timing]    hi-landmarker (2dfan4) x{len(faces)} = {(_t.perf_counter()-t0)*1000:.0f}ms")
    return faces
