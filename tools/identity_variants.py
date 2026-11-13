"""identity_check.py with another way of building inswapper's source vector.

    IDV=<variant> python tools/identity_variants.py --out DIR --boosts 0,0.1,0.25,0.5,1

Measurement only (13 Nov): the app is not changed; Run is monkeypatched.

Variants (w = boost):
  current   e = norm(s + w (s - t))                       (the app)
  ortho     e = norm(s - w t_perp), t_perp = t - (t.s) s   (push only orthogonal to s)
  latent    l = norm(l_s + w (l_s - l_t)) in emap space
  feedback  o = w600k of the swap made with s; e = norm(s + w (s - o))
            (push away from what inswapper actually produced, not from the target)
  feedback2 two feedback steps (second from the first step's result)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
os.chdir(ROOT)

import unleashed.globals as G
from unleashed.processors import FaceSwapInsightFace as M

VARIANT = os.environ.get('IDV', 'current')
_orig_latent = M.FaceSwapInsightFace.identity_latent


def norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def tgt_vec(target_face):
    t = target_face.get('identity_ref')
    if t is None:
        t = target_face.get('embedding')
    return norm(np.asarray(t, np.float32).reshape(1, -1))


def latent_of(self, e):
    l = np.dot(e, self.emap)
    return (l / np.linalg.norm(l)).astype(np.float32)


def raw_run(self, latent, frame):
    b = self.model_swap_insightface.io_binding()
    b.bind_cpu_input('target', frame)
    b.bind_cpu_input('source', latent)
    b.bind_output('output', 'cpu')
    self.model_swap_insightface.run_with_iobinding(b)
    return b.copy_outputs_to_cpu()[0][0]


_rec = None


def embed_crop(out):
    """w600k embedding of inswapper's 128 output (CHW RGB 0..1): arcface 112 = [0:112, 8:120]."""
    global _rec
    if _rec is None:
        from unleashed.face_util import get_face_analyser
        _rec = get_face_analyser().models['recognition']
    img = np.clip(out.transpose(1, 2, 0)[:, :, ::-1] * 255.0, 0, 255).round().astype(np.uint8)
    img = np.ascontiguousarray(img[0:112, 8:120])
    return norm(np.asarray(_rec.get_feat(img), np.float32).reshape(1, -1))


def run(self, source_face, target_face, temp_frame):
    w = float(G.identity_strength or 0.0)
    s = norm(np.asarray(source_face.normed_embedding, np.float32).reshape(1, -1))
    if w <= 0 or VARIANT == 'current':
        return raw_run(self, _orig_latent(self, source_face, target_face), temp_frame)
    if VARIANT == 'ortho':
        t = tgt_vec(target_face)
        tp = t - float(t @ s.T) * s
        return raw_run(self, latent_of(self, norm(s - w * tp)), temp_frame)
    if VARIANT == 'latent':
        t = tgt_vec(target_face)
        ls, lt = latent_of(self, s), latent_of(self, t)
        l = norm(ls + w * (ls - lt)).astype(np.float32)
        return raw_run(self, l, temp_frame)
    if VARIANT.startswith('feedback'):
        e = s
        steps = 2 if VARIANT == 'feedback2' else 1
        for _ in range(steps):
            o = embed_crop(raw_run(self, latent_of(self, e), temp_frame))
            e = norm(e + w * (s - o))
        return raw_run(self, latent_of(self, e), temp_frame)
    raise SystemExit('unknown IDV ' + VARIANT)


M.FaceSwapInsightFace.Run = run

import identity_check
identity_check.main()
