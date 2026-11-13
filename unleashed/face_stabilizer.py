import numpy as np
from threading import Lock


class LandmarkStabilizer:
    """Reduces frame-to-frame landmark jitter in video.

    Per detected face we keep a small track (matched by bbox centre / size) and
    blend the current landmarks with the track's smoothed history using an
    *adaptive* weight: when the face is nearly still we smooth hard (kills
    jitter), when it moves fast we follow the current detection (no lag). This
    is an EMA with motion-gated alpha -- a lightweight One-Euro-style filter
    that is tolerant of frames arriving slightly out of order.

    Smoothing is applied to kps (5x2), landmark_2d_106 and the xy of
    landmark_3d_68 (z is left untouched). Identity/embedding are never changed.
    """

    def __init__(self, strength: float = 0.7, motion_frac: float = 0.05,
                 match_frac: float = 0.6, max_age: int = 8,
                 deadzone_frac: float = 0.0):
        # strength in [0,1]; higher = smoother. Maps to a floor on alpha.
        self.alpha_min = float(np.clip(1.0 - strength, 0.08, 1.0))
        self.motion_frac = motion_frac
        self.match_frac = match_frac
        self.max_age = max_age
        # soft dead-zone: keypoint motion below deadzone_frac*face_size is treated
        # as detector noise and absorbed (alpha -> 0, i.e. the smoothed landmarks
        # freeze), which kills the "landmarks wobble while the head is still"
        # jitter. Real motion above the dead-zone is followed as before. 0 = off
        # (identical to the previous behaviour).
        self.deadzone_frac = max(0.0, float(deadzone_frac))
        self.tracks = []
        self._lock = Lock()

    def reset(self):
        with self._lock:
            self.tracks = []

    def _match(self, center, size):
        best, best_d = None, 1e18
        for t in self.tracks:
            d = float(np.hypot(*(center - t['center'])))
            if d < self.match_frac * size and d < best_d:
                best_d, best = d, t
        return best

    @staticmethod
    def _get(face, key):
        try:
            v = face[key]
        except Exception:
            v = getattr(face, key, None)
        return None if v is None else np.asarray(v)

    def stabilize(self, faces):
        if not faces:
            return faces
        with self._lock:
            for t in self.tracks:
                t['age'] += 1
            self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

            for f in faces:
                bbox = np.asarray(f.bbox, dtype=np.float32)
                size = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])) + 1e-6
                center = np.array([(bbox[0] + bbox[2]) * 0.5,
                                   (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)

                kps = self._get(f, 'kps')
                lm106 = self._get(f, 'landmark_2d_106')
                lm68 = self._get(f, 'landmark_3d_68')

                t = self._match(center, size)
                if t is None:
                    self.tracks.append({
                        'center': center, 'size': size, 'age': 0,
                        'kps': None if kps is None else kps.copy(),
                        'lm106': None if lm106 is None else lm106.copy(),
                        'lm68': None if lm68 is None else lm68.copy(),
                    })
                    continue

                # adaptive alpha from how far the keypoints moved
                if kps is not None and t['kps'] is not None and t['kps'].shape == kps.shape:
                    motion = float(np.linalg.norm(kps - t['kps'], axis=1).mean())
                else:
                    motion = float(np.hypot(*(center - t['center'])))
                # soft dead-zone. Inside the zone (motion <= dead) the smoothed
                # landmarks freeze (alpha=0) -> still-head detector wobble is
                # killed. Just outside, the alpha floor ramps 0 -> alpha_min across
                # (dead, 2*dead] so slow real motion is picked up without a jump.
                # deadzone_frac=0 -> dead=0 -> falls straight to the original
                # clip(motion/(motion_frac*size), alpha_min, 1) formula.
                dead = self.deadzone_frac * size
                if dead > 0.0 and motion <= dead:
                    alpha = 0.0
                else:
                    motion_eff = motion - dead if dead > 0.0 else motion
                    raw = motion_eff / (self.motion_frac * size)
                    if dead > 0.0:
                        floor = self.alpha_min * min((motion - dead) / dead, 1.0)
                    else:
                        floor = self.alpha_min
                    alpha = float(np.clip(raw, floor, 1.0))
                beta = 1.0 - alpha

                def smooth(cur, prev):
                    if cur is None:
                        return None, prev
                    if prev is None or prev.shape != cur.shape:
                        return cur, cur.copy()
                    out = (alpha * cur + beta * prev).astype(cur.dtype)
                    return out, out.copy()

                new_kps, t['kps'] = smooth(kps, t['kps'])
                new_lm106, t['lm106'] = smooth(lm106, t['lm106'])

                new_lm68 = lm68
                if lm68 is not None:
                    if t['lm68'] is None or t['lm68'].shape != lm68.shape:
                        t['lm68'] = lm68.copy()
                    else:
                        new_lm68 = lm68.copy()
                        new_lm68[:, :2] = (alpha * lm68[:, :2] +
                                           beta * t['lm68'][:, :2])
                        t['lm68'] = new_lm68.copy()

                t['center'], t['size'], t['age'] = center, size, 0

                # write smoothed landmarks back onto the face object
                try:
                    if new_kps is not None:
                        f['kps'] = new_kps
                    if new_lm106 is not None:
                        f['landmark_2d_106'] = new_lm106
                    if new_lm68 is not None:
                        f['landmark_3d_68'] = new_lm68
                except Exception:
                    pass
        return faces



class TargetTracks:
    """Per-face running average over a video of the target's face shape in the
    reference frame (Source jaw & chin). It describes the PERSON, not the frame,
    so averaging it along the face's track keeps the effect from flickering.
    The identity embedding is tracked too, only to notice a cut to another
    person (the track then starts over).

    Separate from LandmarkStabilizer on purpose: it never touches landmarks,
    and it runs whether landmark smoothing is on or not. Must be fed frames in
    order (it is called inside the frame sequencer's ordered section).

    Writes face['shape_canon'] and face['shape_gate'].
    """

    def __init__(self, match_frac: float = 0.6, max_age: int = 8,
                 min_rate: float = 0.05, cut_similarity: float = 0.5):
        self.match_frac = match_frac
        self.max_age = max_age
        self.min_rate = min_rate              # slowest update (after ~20 frames)
        self.cut_similarity = cut_similarity  # below: another person, restart
        self.tracks = []
        self._lock = Lock()

    def reset(self):
        with self._lock:
            self.tracks = []

    def _match(self, center, size):
        best, best_d = None, 1e18
        for t in self.tracks:
            d = float(np.hypot(*(center - t['center'])))
            if d < self.match_frac * size and d < best_d:
                best_d, best = d, t
        return best

    def update(self, faces, shape=True):
        if not faces:
            return faces
        canonical = landmarks3d = gate_of = None
        if shape:
            from unleashed.face_shape import canonical, gate_of, landmarks3d
        with self._lock:
            for t in self.tracks:
                t['age'] += 1
            self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
            for f in faces:
                bbox = np.asarray(f.bbox, dtype=np.float32)
                size = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])) + 1e-6
                center = np.array([(bbox[0] + bbox[2]) * 0.5,
                                   (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
                emb = LandmarkStabilizer._get(f, 'embedding')
                if emb is not None:
                    emb = emb.astype(np.float32).reshape(-1)
                    emb = emb / (np.linalg.norm(emb) + 1e-9)
                t = self._match(center, size)
                if t is None:
                    t = {'n': 0, 'emb': None, 'shape': None, 'gate': None}
                    self.tracks.append(t)
                elif emb is not None and t['emb'] is not None \
                        and float(t['emb'] @ emb) < self.cut_similarity:
                    t.update(n=0, emb=None, shape=None, gate=None)   # a cut to someone else: start over
                t['center'], t['age'] = center, 0
                t['n'] += 1
                # running mean for the first frames, then a slow EMA
                a = max(1.0 / t['n'], self.min_rate)
                if emb is not None:
                    e = emb if t['emb'] is None else (1.0 - a) * t['emb'] + a * emb
                    t['emb'] = e / (np.linalg.norm(e) + 1e-9)
                if shape:
                    L3 = landmarks3d(f)
                    if L3 is not None:
                        c = canonical(L3)
                        t['shape'] = c if t['shape'] is None else (1.0 - a) * t['shape'] + a * c
                        # the pose fade follows the head, only a little calmer
                        g = gate_of(L3)
                        t['gate'] = g if t['gate'] is None else 0.7 * t['gate'] + 0.3 * g
                try:
                    if shape and t['shape'] is not None:
                        f['shape_canon'] = t['shape'].copy()
                        f['shape_gate'] = t['gate']
                except Exception:
                    pass
        return faces
