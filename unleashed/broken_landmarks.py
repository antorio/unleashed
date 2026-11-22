"""Ignore broken landmarks (video): unleashed.globals.landmark_hold_broken.

A hand waving in front of the face, a shoulder, a hand under the chin: the
landmark models (1k3d68 and 2dfan4 alike) then give points that are right in
one frame, wrong in the next, right again -- and the whole set usually jumps
together (the detector box and kps move too), so the landmarks alone cannot
tell a broken frame from head motion.

Here the head motion comes from the PIXELS instead: sparse optical flow
(Lucas-Kanade, forward-backward checked) of the face area from the previous
frame, a RANSAC similarity on it. Per face track:

  pred = the last kept points moved by that flow
  r    = distance of the new points from pred (5 alignment points, / eye distance)
  r <= THR                 -> new points kept (normal frame)
  r >  THR, but            -> new points kept when
     * the image itself moved with them (each point tracked by flow agrees
       with the new point: median <= VETO) -- a real expression change
       (mouth opening) moves the pixels, an occluder does not move the face
     * or they agree with the previous raw frame for C frames in a row
       (a new steady state, e.g. something stays in front of the face)
     * or K frames were held already
  otherwise                -> the frame is broken: the last kept points moved by
                              the flow are used (kps / 106 / 68 alike)

Runs in the frame sequencer's ordered section, before landmark smoothing.
Measured on a synthetic clip (scratchpad/occ): waving hand, frames that jump
more than 3% of the eye distance 22 -> 2 (1k3d68) / 22 -> 2 (2dfan4); a fast
jaw drop is not held back; clean frames are untouched.
"""
from threading import Lock

import cv2
import numpy as np

from unleashed.face_util import landmark_68_to_5

THR = 0.04          # new points this far from the flow prediction (eye distances) -> suspect
VETO = 0.015        # ... unless the pixels at the points moved with them this closely
STEP = 0.025        # raw vs previous raw (moved): "consistent"
C = 2               # consistent frames in a row -> accepted as the new state
K = 8               # longest hold
MAX_AGE = 8


def _lk():
    return dict(winSize=(15, 15), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class BrokenLandmarkFilter:
    def __init__(self):
        self.tracks = []
        self._lock = Lock()

    def reset(self):
        with self._lock:
            self.tracks = []

    @staticmethod
    def _get(face, key):
        try:
            v = face[key]
        except Exception:
            v = getattr(face, key, None)
        return None if v is None else np.asarray(v)

    @staticmethod
    def _points(face):
        """(points used to judge, 5 alignment points of them, eye distance)"""
        lm68 = BrokenLandmarkFilter._get(face, 'landmark_3d_68')
        if lm68 is not None and lm68.shape[0] == 68:
            P = lm68[:, :2].astype(np.float64)
            iod = float(np.linalg.norm(P[36:42].mean(0) - P[42:48].mean(0)))
            return P, '68', iod
        kps = BrokenLandmarkFilter._get(face, 'kps')
        if kps is None:
            return None, None, 0.0
        P = kps[:, :2].astype(np.float64)
        return P, 'kps', float(np.linalg.norm(P[1] - P[0]))

    @staticmethod
    def _five(P, kind):
        if kind == '68':
            return landmark_68_to_5(np.concatenate([P, np.zeros((len(P), 1))], 1))
        return P

    def _drop(self, t):
        # by identity: list.remove compares the dicts, i.e. their numpy arrays
        # (ValueError as soon as there are two tracks)
        self.tracks = [x for x in self.tracks if x is not t]

    def _match(self, center, size):
        best, best_d = None, 1e18
        for t in self.tracks:
            if t['age'] == 0:               # already taken by a face of this frame
                continue
            d = float(np.hypot(*(center - t['center'])))
            if d < 0.6 * size and d < best_d:
                best_d, best = d, t
        return best

    @staticmethod
    def _roi(P, size, shape):
        h, w = shape[:2]
        x0, y0 = P.min(0) - 0.6 * size
        x1, y1 = P.max(0) + 0.6 * size
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(w, x1)), int(min(h, y1))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _flow(g0, g1, S, iod):
        """Similarity of the face area g0 -> g1 (ROI coordinates) or None."""
        hull = cv2.convexHull(np.round(S).astype(np.int32))
        m = np.zeros_like(g0)
        cv2.fillConvexPoly(m, hull, 255)
        k = max(3, int(0.3 * iod) | 1)
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
        p0 = cv2.goodFeaturesToTrack(g0, 150, 0.01, max(3.0, 0.06 * iod), mask=m)
        if p0 is None or len(p0) < 8:
            return None
        lk = _lk()
        lk['winSize'] = (21, 21)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((pb - p0)[:, 0], axis=1) < 0.5)
        if ok.sum() < 8:
            return None
        # RANSAC draws from OpenCV's per-thread RNG: without a fixed seed the
        # result depended on which worker thread ran the frame (8 threads !=
        # 1 thread on a 6-face clip)
        cv2.setRNGSeed(1234)
        T, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                             ransacReprojThreshold=max(1.0, 0.02 * iod))
        if T is None or inl is None or inl.sum() < 6:
            return None
        return T

    @staticmethod
    def _pixels_agree(g0, g1, S, P, iod):
        """The image at the kept points moved to the new points (median <= VETO)."""
        p0 = S.astype(np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **_lk())
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **_lk())
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((pb - p0)[:, 0], axis=1) < 0.7)
        if ok.sum() < max(4, len(S) // 3):
            return False
        return float(np.median(np.linalg.norm(P[ok] - p1[ok, 0], axis=1))) / iod <= VETO

    def filter(self, faces, frame):
        """faces: this frame's faces, in frame order (ordered section). Writes the
        held points back onto broken faces. Returns the number of faces held."""
        if not faces or frame is None:
            return 0
        held = 0
        with self._lock:
            for t in self.tracks:
                t['age'] += 1
            self.tracks = [t for t in self.tracks if t['age'] <= MAX_AGE]
            shape = frame.shape
            crop = lambda r: _gray(frame[r[1]:r[3], r[0]:r[2]])    # grey of the face area only
            for f in faces:
                P, kind, iod = self._points(f)
                if P is None or iod < 4.0:
                    continue
                bbox = np.asarray(f.bbox, dtype=np.float64)
                size = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])) + 1e-6
                center = (bbox[:2] + bbox[2:4]) * 0.5
                t = self._match(center, size)
                roi = self._roi(P if t is None else np.vstack([P, t['S']]), size, shape)
                state = dict(center=center, size=size, age=0, kind=kind, raw=P.copy(), bad=0, cons=0,
                             S=P.copy(), lm68=self._get(f, 'landmark_3d_68'), kps=self._get(f, 'kps'),
                             lm106=self._get(f, 'landmark_2d_106'), roi=roi,
                             g=None if roi is None else crop(roi))
                if t is None or t['kind'] != kind or t['age'] != 1 or t['g'] is None or t['roi'] is None:
                    # new face, a gap, or a different point set: start over here
                    if t is not None:
                        self._drop(t)
                    self.tracks.append(state)
                    continue
                x0, y0, x1, y1 = t['roi']
                g0, g1 = t['g'], crop(t['roi'])
                off = np.array([x0, y0], np.float64)
                S = t['S']
                T = self._flow(g0, g1, S - off, iod)
                if T is None:
                    self._drop(t)
                    self.tracks.append(state)
                    continue
                A, b = T[:, :2], T[:, 2]
                mv = lambda X: (X - off) @ A.T + b + off
                pred = mv(S)
                r = np.linalg.norm(self._five(P, kind) - self._five(pred, kind), axis=1).mean() / iod
                q = np.linalg.norm(self._five(P, kind) - self._five(mv(t['raw']), kind), axis=1).mean() / iod
                cons = t['cons'] + 1 if q <= STEP else 0
                keep = (r <= THR or cons >= C or t['bad'] >= K
                        or self._pixels_agree(g0, g1, S - off, P - off, iod))
                if keep:
                    state['cons'] = cons
                    self._drop(t)
                    self.tracks.append(state)
                    continue
                # broken frame: the kept points, moved by the image motion
                held += 1
                new = dict(state, bad=t['bad'] + 1, cons=cons, S=pred)
                try:
                    if kind == '68' and t['lm68'] is not None:
                        lm = np.array(t['lm68'], dtype=np.float32, copy=True)
                        lm[:, :2] = mv(lm[:, :2].astype(np.float64))
                        f['landmark_3d_68'] = lm
                        new['lm68'] = lm
                    if t['kps'] is not None:
                        k5 = mv(np.asarray(t['kps'], np.float64)).astype(np.float32)
                        f['kps'] = k5
                        new['kps'] = k5
                    if t['lm106'] is not None:
                        l106 = mv(np.asarray(t['lm106'], np.float64)).astype(np.float32)
                        f['landmark_2d_106'] = l106
                        new['lm106'] = l106
                except Exception:
                    pass
                self._drop(t)
                self.tracks.append(new)
        return held
