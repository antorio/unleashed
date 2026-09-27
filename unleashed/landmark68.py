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

    def detect(self, frame_bgr, bbox, angle=0.0):
        """angle (degrees, cv2 sense): the crop is turned by it about the box
        centre first, e.g. the eye line's angle so the face reaches the model
        level. 0: exactly FaceFusion's crop."""
        b = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
        wh = float(np.maximum(np.max(b[2:] - b[:2]), 1.0))
        scale = 195.0 / wh
        tx, ty = (self.MODEL_SIZE - (b[2:] + b[:2]) * scale) * 0.5
        affine = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
        if angle:
            centre = (float(b[0] + b[2]) * 0.5, float(b[1] + b[3]) * 0.5)
            R = np.vstack([cv2.getRotationMatrix2D(centre, float(angle), 1.0), [0.0, 0.0, 1.0]])
            affine = (affine.astype(np.float64) @ R).astype(np.float32)

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


def _level_angle(face, lm68):
    """The eye line's angle (degrees): the face is turned level before 2dfan4.
    From buffalo's 68 points (the landmarks that stay put best when the head
    tilts); the detector's 5 points when they are missing."""
    try:
        if lm68 is not None:
            p = np.asarray(lm68, dtype=np.float64)[:, :2]
            left, right = p[36:42].mean(0), p[42:48].mean(0)
        else:
            k = np.asarray(face['kps'], dtype=np.float64)
            left, right = k[0], k[1]
        d = right - left
        return float(np.degrees(np.arctan2(d[1], d[0])))
    except Exception:
        return 0.0


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
    min_score = float(getattr(unleashed.globals, 'hi_landmarker_min_score', 0.0) or 0.0)
    level = bool(getattr(unleashed.globals, 'hi_landmarker_level', False))
    frontal = bool(getattr(unleashed.globals, 'hi_landmarker_frontal_only', False))
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
            prev = f.get('landmark_3d_68') if hasattr(f, 'get') else None
            angle = _level_angle(f, prev) if level else 0.0
            pts, score = lm.detect(frame, bbox, angle)
            if pts is None or pts.shape[0] != 68:
                continue
            # 2dfan4's confidence (like FaceFusion's face_landmarker_score):
            # below min_score its points are replaced by buffalo's 68 (blended
            # over +-0.1 around it, so a score near the line does not flip the
            # alignment between the two sets, which sit ~7% of the eye distance
            # apart). Measured on a neutral photo: clean faces score 0.91-0.97;
            # with a 4 px blur 2dfan4 fails outright (median error 1.07 x the eye
            # distance, buffalo 0.17) and scores 0.15; below 0.3 it was off by
            # 1.17 in median against buffalo's 0.25.
            weight = 1.0
            if min_score > 0.0 and prev is not None:
                weight = float(np.clip((score - min_score) / 0.2 + 0.5, 0.0, 1.0))
            # frontal only: fade to buffalo's 68 as the head turns up / down
            # (pitch 10 -> 25 deg) or sideways (yaw 30 -> 50 deg), buffalo's
            # pose. Measured (neutral photo): on two faces looking up (pitch
            # -20/-22) 2dfan4's eye-to-mouth distance was 4-7% shorter than
            # buffalo's (~0% on level faces) -- the swap came out shorter, the
            # "face gets smaller when looking up"; and its landmarks moved ~1.6x
            # more than buffalo's under tilt / blur at |yaw| 45-90.
            if frontal and prev is not None:
                pose = f.get('pose') if hasattr(f, 'get') else None
                if pose is not None and len(pose) >= 2:
                    fade = lambda x, a, b: float(np.clip((b - abs(float(x))) / (b - a), 0.0, 1.0))
                    weight *= min(fade(pose[0], 10.0, 25.0), fade(pose[1], 30.0, 50.0))
            if weight <= 0.0:
                if dbg:
                    print(f"[hi-landmarker] score {score:.2f} (min {min_score:.2f}) / pose: buffalo's 68 kept")
                continue
            if weight < 1.0:
                pts = weight * pts + (1.0 - weight) * np.asarray(prev, dtype=np.float32)[:, :2]
            # z is filled with zeros: 2dfan4 is a 2D landmarker, and nothing that
            # reads landmark_3d_68 uses the z column (landmark_68_to_5 takes
            # [:, :2] and the stabilizer leaves z untouched). buffalo_l's 1k3d68
            # runs alongside 2dfan4 only for face shape (kept below).
            z = np.zeros((68, 1), dtype=np.float32)
            # face shape from the source compares 3D shapes: it asks for
            # buffalo's 68 points (with depth) and reads this copy of them
            if prev is not None:
                f['landmark_3d_68_buffalo'] = prev
            f['landmark_3d_68'] = np.concatenate([pts.astype(np.float32), z], axis=1)
            if dbg:
                print(f"[hi-landmarker] refined 68pts (score={score:.2f}, weight={weight:.2f}, angle={angle:.1f})")
        except Exception as e:
            if dbg:
                print(f"[hi-landmarker] per-face refine failed: {e}")

    if prof and t0 is not None:
        import time as _t
        print(f"[timing]    hi-landmarker (2dfan4) x{len(faces)} = {(_t.perf_counter()-t0)*1000:.0f}ms")
    return faces
