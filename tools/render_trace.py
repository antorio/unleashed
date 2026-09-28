"""Which step makes a single frame come out different between two renders?

    python tools/render_trace.py        (stop the Unleashed server first)

repeat_render showed it on the L4 with ER off: two renders of the same clip
matched except 3 of 24 frames, each far off (max 104-134 levels, ~10 000
pixels), their neighbours identical. This renders repeat_render's clip with
ER off, four times one after another and three at once, and fingerprints
every step of every face: detection, the frame-order turn (and whether it
ran out of time), landmark smoothing, alignment, each swap slice, the
occlusion mask, the paste. For every frame that differs from the first
render it names the first step that differs and whether that step got the
same input (the step itself is not repeatable) or a different one.
Writes only to --out (/content/unleashed_test on Colab).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('repeat_render', os.path.join(HERE, 'repeat_render.py'))
RR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RR)
ROOT = RR.ROOT
SEQUENTIAL = ['run1', 'run2', 'run3', 'run4']
AT_ONCE = ['together1', 'together2', 'together3']
STEPS = ['detect', 'smooth', 'align', 'swap', 'mask', 'paste']


def fp(*arrays):
    h = hashlib.md5()
    for a in arrays:
        if a is None:
            h.update(b'None')
            continue
        a = np.ascontiguousarray(np.asarray(a))
        h.update(repr((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:12]


def face_fp(f):
    return fp(f.bbox, f.kps, getattr(f, 'landmark_2d_106', None), getattr(f, 'landmark_3d_68', None))


def child(label, er, out):
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    work = os.path.join(out, label)
    os.makedirs(work, exist_ok=True)
    import unleashed.globals as G
    from settings import Settings
    G.CFG = Settings('config.yaml')                          # never saved back
    G.CFG.config_file = os.path.join(work, 'config_copy.yaml')
    G.CFG.output_folder = G.output_path = work
    from unleashed import core
    import unleashed.ProcessMgr as PM
    import unleashed.face_util as FU
    from unleashed.face_stabilizer import LandmarkStabilizer
    from unleashed.processors.FaceSwapInsightFace import FaceSwapInsightFace
    from unleashed.ProcessEntry import ProcessEntry
    from ui.tabs import faceswap_state as S
    G.execution_providers = core.decode_execution_providers([G.CFG.provider])
    G.video_encoder, G.video_quality = G.CFG.output_video_codec, G.CFG.video_quality

    tl = threading.local()
    log, frames, turns = [], {}, {}
    lock = threading.Lock()

    def note(step, inp, outp):
        k = tl.k.get(step, 0) if hasattr(tl, 'k') else 0
        if hasattr(tl, 'k'):
            tl.k[step] = k + 1
        with lock:
            log.append({'frame': getattr(tl, 'frame', None), 'face': getattr(tl, 'face', None), 'step': step, 'k': k,
                        'in': inp, 'out': outp})

    o_pf, o_face, o_align_r, o_align = PM.ProcessMgr.process_frame, PM.ProcessMgr.process_face, FU.align_crop_robust, FU.align_crop
    o_detect_all, o_detect_first, o_stab = PM.get_all_faces_multi, PM.get_first_face_multi, LandmarkStabilizer.stabilize
    o_swap, o_mask, o_paste, o_turn = FaceSwapInsightFace.Run, PM.ProcessMgr.compute_mask, PM.ProcessMgr.paste_upscale, PM.FrameSequencer.in_order

    def pf(self, frame, frame_index=None):
        tl.frame, tl.face, tl.k = frame_index, None, {}
        result = o_pf(self, frame, frame_index)
        frames[frame_index] = None if result is None else result.copy()
        return result

    def detect_all(frame, *a, **k):
        faces = o_detect_all(frame, *a, **k)
        note('detect', fp(frame), fp(*[face_fp(f) for f in sorted(faces or [], key=lambda f: f.bbox[0])]))
        return faces

    def detect_first(frame, *a, **k):
        face = o_detect_first(frame, *a, **k)
        note('detect', fp(frame), face_fp(face) if face is not None else 'none')
        return face

    @contextmanager
    def turn(self, index, timeout=10.0):
        t0 = time.time()
        with o_turn(self, index, timeout) as x:
            with lock:
                turns[index] = {'wait': round(time.time() - t0, 3), 'early': index is not None and self._expected < index}
            yield x

    def stab(self, faces):
        before = fp(*[face_fp(f) for f in faces])
        r = o_stab(self, faces)
        note('smooth', before, fp(*[face_fp(f) for f in faces]))
        return r

    def face(self, face_index, target_face, frame):
        tl.face, tl.k = int(target_face.bbox[0]), {}
        return o_face(self, face_index, target_face, frame)

    def align_r(img, lmk5, size=128):
        crop, M = o_align_r(img, lmk5, size)
        note('align', fp(img, lmk5), fp(crop, M))
        return crop, M

    def align(img, lmk, size=112, mode='arcface'):
        crop, M = o_align(img, lmk, size, mode)
        note('align', fp(img, lmk), fp(crop, M))
        return crop, M

    def swap(self, source_face, target_face, temp_frame):
        r = o_swap(self, source_face, target_face, temp_frame)
        note('swap', fp(temp_frame, getattr(source_face, 'embedding', None)), fp(r))
        return r

    def mask(self, processor, frame):
        r = o_mask(self, processor, frame)
        note('mask', fp(frame), fp(r))
        return r

    def paste(self, fake_face, upsk_face, M, target_img, *a, **k):
        inp = fp(fake_face, upsk_face, M, target_img)
        r = o_paste(self, fake_face, upsk_face, M, target_img, *a, **k)
        note('paste', inp, fp(r))
        return r

    PM.ProcessMgr.process_frame, PM.ProcessMgr.process_face = pf, face
    FU.align_crop_robust, FU.align_crop = align_r, align           # process_face imports them from face_util per call
    PM.get_all_faces_multi, PM.get_first_face_multi = detect_all, detect_first
    LandmarkStabilizer.stabilize, FaceSwapInsightFace.Run = stab, swap
    PM.ProcessMgr.compute_mask, PM.ProcessMgr.paste_upscale, PM.FrameSequencer.in_order = mask, paste, turn

    clip, source = RR.make_inputs(out)
    S.add_sources([source])
    S.values.update(S.FACTORY)
    S.values.update({'mode': 'All faces', 'er': bool(er)})
    S.apply_settings()
    v = dict(S.values)
    plugin, _ = S.mask_plugin()
    core.batch_process_regular(S.SWAP_MODEL, 'File', [ProcessEntry(clip, 0, 0, 0)], plugin, v['mask_objects'],
                               True, None, bool(v['keep_mouth']), bool(v['keep_eyes']), int(v['passes']), None,
                               S.active_source_index())
    np.savez_compressed(os.path.join(out, label + '.npz'), **{str(i): f for i, f in frames.items() if f is not None})
    json.dump({'log': log, 'turns': {str(k): v for k, v in turns.items()}}, open(os.path.join(out, label + '_trace.json'), 'w'))
    print(f'{label}: {len(frames)} frames, {len(log)} steps, threads {G.execution_threads}', flush=True)
    os._exit(0)


def compare(label, reference, out):
    fa, fb = np.load(os.path.join(out, reference + '.npz')), np.load(os.path.join(out, label + '.npz'))
    ta, tb = (json.load(open(os.path.join(out, x + '_trace.json'))) for x in (reference, label))
    key = lambda s: (s['frame'], s['face'] if s['face'] is not None else -1, STEPS.index(s['step']), s['k'])
    sa = {key(s): s for s in ta['log']}
    sb = {key(s): s for s in tb['log']}
    early = sorted(int(k) for k, t in tb['turns'].items() if k != 'None' and t['early'])
    slow = sorted(((int(k), t['wait']) for k, t in tb['turns'].items() if k != 'None' and t['wait'] > 2), key=lambda x: -x[1])[:3]
    notes = []
    if early:
        notes.append(f'frames that ran out of time for their turn and went in early: {early}')
    if slow:
        notes.append('longest waits for the turn: ' + ', '.join(f'frame {f} {w:.1f} s' for f, w in slow))
    differ = []
    for k in sorted(set(fa.files) & set(fb.files), key=int):
        if np.array_equal(fa[k], fb[k]):
            continue
        d = np.abs(fa[k].astype(np.int16) - fb[k].astype(np.int16)).max(axis=2)
        i = int(k)
        where = 'no step differs (after the paste?)'
        for sk in sorted(k2 for k2 in sa if k2[0] == i):
            a, b = sa[sk], sb.get(sk)
            if b is None:
                where = f'{a["step"]} (face at x={a["face"]}) missing in this run'
                break
            if a['in'] != b['in'] or a['out'] != b['out']:
                how = 'same input, other output' if a['in'] == b['in'] else 'other input'
                where = f'{a["step"]} #{a["k"]} (face at x={a["face"]}): {how}'
                break
        differ.append(f'frame {i} (max {int(d.max())}, {int((d > 2).sum())} px >2) first differs at {where}')
    print(f'  {label} vs {reference}: ' + ('identical' if not differ else f'{len(differ)} frames differ'), flush=True)
    for line in differ + notes:
        print(f'    {line}', flush=True)
    return not differ


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='/content/unleashed_test' if os.path.isdir('/content') else os.path.join(ROOT, 'temp', 'repeat_render'))
    parser.add_argument('--child')
    parser.add_argument('--er', type=int, default=0)
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if args.child:
        return child(args.child, args.er, out)

    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from unleashed import core
    core.pre_check()
    RR.make_inputs(out)
    env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE='1', PYTHONUNBUFFERED='1')

    def start(label):
        log = open(os.path.join(out, label + '.log'), 'w')
        return subprocess.Popen([sys.executable, os.path.abspath(__file__), '--child', label, '--er', str(args.er),
                                 '--out', out], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)

    ok = {}
    print(f'ER {"on" if args.er else "off"}; one after another:', flush=True)
    for label in SEQUENTIAL:
        t0 = time.time()
        ok[label] = start(label).wait() == 0
        print(f'  {label}: {"done" if ok[label] else f"FAILED (see {label}.log)"} in {time.time() - t0:.0f} s', flush=True)
    print('three at once:', flush=True)
    t0 = time.time()
    procs = [(label, start(label)) for label in AT_ONCE]
    for label, proc in procs:
        ok[label] = proc.wait() == 0
        print(f'  {label}: {"done" if ok[label] else f"FAILED (see {label}.log)"} in {time.time() - t0:.0f} s', flush=True)
    print('compared with run1:', flush=True)
    bad = [label for label in SEQUENTIAL[1:] + AT_ONCE if ok.get(label) and ok.get('run1') and not compare(label, 'run1', out)]
    print(f'RESULT: {len(bad)} of {len(SEQUENTIAL) - 1 + len(AT_ONCE)} renders differ from run1', flush=True)


if __name__ == '__main__':
    main()
