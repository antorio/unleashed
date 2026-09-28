"""Where does ER make two renders differ? (the follow-up to repeat_render.py)

    python tools/er_trace.py            (stop the Unleashed server first)

Renders repeat_render's neutral clip with ER on, twice per setup, each run in
its own process. Every LivePortrait model call is logged with its frame,
face, model, and a fingerprint of its inputs and outputs. Setups:

    default        the app as it is (cuDNN EXHAUSTIVE algorithm search)
    deterministic  ONNX Runtime's use_deterministic_compute on every session
    heuristic      cuDNN HEURISTIC algorithm search

For each setup it prints how the two renders differ and where they part:
"same inputs, different outputs" means the model itself does not repeat;
"different inputs" means something before it already differed. Writes only
to --out (/content/unleashed_test on Colab).
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('repeat_render', os.path.join(HERE, 'repeat_render.py'))
RR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RR)
ROOT = RR.ROOT
SETUPS = ['default', 'deterministic', 'heuristic']
MODELS = ('feature_extractor', 'motion_extractor', 'generator', 'stitcher')


def fingerprint(arrays):
    h = hashlib.md5()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(repr((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:12]


def child(label, setup, out):
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    work = os.path.join(out, label)
    os.makedirs(work, exist_ok=True)
    import onnxruntime as ort
    if setup == 'deterministic':
        base = ort.InferenceSession

        class Deterministic(base):
            def __init__(self, path, sess_options=None, *args, **kwargs):
                options = sess_options or ort.SessionOptions()
                options.use_deterministic_compute = True
                super().__init__(path, options, *args, **kwargs)
        ort.InferenceSession = Deterministic
    import unleashed.globals as G
    if setup == 'heuristic':
        G.cudnn_conv_algo_search = 'HEURISTIC'
    from settings import Settings
    G.CFG = Settings('config.yaml')                          # never saved back
    G.CFG.config_file = os.path.join(work, 'config_copy.yaml')
    G.CFG.output_folder = G.output_path = work
    from unleashed import core
    import unleashed.ProcessMgr as PM
    from unleashed.ProcessEntry import ProcessEntry
    from unleashed.processors.Expression_LivePortrait import Expression_LivePortrait as LP
    from ui.tabs import faceswap_state as S
    G.execution_providers = core.decode_execution_providers([G.CFG.provider])
    G.video_encoder, G.video_quality = G.CFG.output_video_codec, G.CFG.video_quality

    tl = threading.local()
    frames, calls, small = {}, [], {}
    lock = threading.Lock()
    process_frame, process_expression, run_session = PM.ProcessMgr.process_frame, PM.ProcessMgr.process_expression, LP._run_session

    def pf(self, frame, frame_index=None):
        tl.frame = frame_index
        result = process_frame(self, frame, frame_index)
        frames[frame_index] = None if result is None else result.copy()
        return result

    def pe(self, processor, aligned_img, fake_frame, frame=None, target_face=None):
        tl.face = [int(v) for v in target_face.bbox[:2]] if target_face is not None else None
        tl.k = {}
        tl.crop = [fingerprint([aligned_img]), fingerprint([fake_frame]), fingerprint([frame])]
        return process_expression(self, processor, aligned_img, fake_frame, frame, target_face)

    def rs(self, session, feeds):
        name = next((m for m in MODELS if getattr(self, m, None) is session), '?')
        k = tl.k.get(name, 0)
        tl.k[name] = k + 1
        result = run_session(self, session, feeds)
        key = f"{getattr(tl, 'frame', None)}|{getattr(tl, 'face', None)}|{name}|{k}"
        rec = {'key': key, 'frame': getattr(tl, 'frame', None), 'model': name, 'crop': getattr(tl, 'crop', None),
               'in': fingerprint([feeds[n] for n in sorted(feeds)]), 'out': fingerprint(result)}
        with lock:
            calls.append(rec)
            if name == 'motion_extractor':
                small[key] = np.concatenate([np.asarray(r, np.float32).reshape(-1) for r in result[:6]])
        return result

    PM.ProcessMgr.process_frame, PM.ProcessMgr.process_expression, LP._run_session = pf, pe, rs
    clip, source = RR.make_inputs(out)
    S.add_sources([source])
    S.values.update(S.FACTORY)
    S.values.update({'mode': 'All faces', 'er': True})
    S.apply_settings()
    v = dict(S.values)
    plugin, _ = S.mask_plugin()
    core.batch_process_regular(S.SWAP_MODEL, 'File', [ProcessEntry(clip, 0, 0, 0)], plugin, v['mask_objects'],
                               True, None, bool(v['keep_mouth']), bool(v['keep_eyes']), int(v['passes']), None,
                               S.active_source_index())
    np.savez_compressed(os.path.join(out, label + '.npz'), **{str(i): f for i, f in frames.items() if f is not None})
    np.savez_compressed(os.path.join(out, label + '_motion.npz'), **small)
    json.dump(calls, open(os.path.join(out, label + '_calls.json'), 'w'))
    print(f'{label}: {len(frames)} frames, {len(calls)} model calls', flush=True)
    os._exit(0)


def compare(a_label, b_label, out):
    """How two runs of the same setup differ, and where they part first."""
    fa, fb = np.load(os.path.join(out, a_label + '.npz')), np.load(os.path.join(out, b_label + '.npz'))
    big, small_d = [], 0
    for k in sorted(set(fa.files) & set(fb.files), key=int):
        d = np.abs(fa[k].astype(np.int16) - fb[k].astype(np.int16)).max(axis=2)
        if int(d.max()) > 20:
            big.append(f'{k} (max {int(d.max())}, {int((d > 2).sum())} px >2)')
        elif d.any():
            small_d += 1
    ca = {c['key']: c for c in json.load(open(os.path.join(out, a_label + '_calls.json')))}
    cb = {c['key']: c for c in json.load(open(os.path.join(out, b_label + '_calls.json')))}
    ma, mb = np.load(os.path.join(out, a_label + '_motion.npz')), np.load(os.path.join(out, b_label + '_motion.npz'))
    per = {m: [0, 0, 0] for m in MODELS}           # same, same inputs + other outputs, other inputs
    first = None
    order = sorted(set(ca) & set(cb), key=lambda k: (int(k.split('|')[0]) if k.split('|')[0] != 'None' else -1, k))
    worst = (0.0, None)
    for key in order:
        a, b = ca[key], cb[key]
        m = a['model']
        if a['in'] != b['in']:
            per.setdefault(m, [0, 0, 0])[2] += 1
            kind = 'different inputs' + (' (the crops going in already differ)' if a['crop'] != b['crop'] else '')
        elif a['out'] != b['out']:
            per.setdefault(m, [0, 0, 0])[1] += 1
            kind = 'same inputs, different outputs'
        else:
            per.setdefault(m, [0, 0, 0])[0] += 1
            continue
        if m == 'motion_extractor' and key in ma.files and key in mb.files:
            delta = float(np.abs(ma[key] - mb[key]).max())
            kind += f' (max |d| {delta:.2g})'
            if delta > worst[0]:
                worst = (delta, key)
        if first is None:
            first = f'frame {a["frame"]}, {m}: {kind}'
    print(f'  frames: {len(big)} with a big difference ({", ".join(big[:5])}{" ..." if len(big) > 5 else ""}), '
          f'{small_d} with a small one', flush=True)
    print(f'  first model call that differs: {first or "none"}', flush=True)
    for m, (same, outs, ins) in per.items():
        if same or outs or ins:
            print(f'    {m}: {same} same, {outs} same inputs but other outputs, {ins} other inputs', flush=True)
    if worst[1]:
        print(f'    largest motion difference: {worst[0]:.3g} at {worst[1]}', flush=True)
    return not big and not small_d


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='/content/unleashed_test' if os.path.isdir('/content') else os.path.join(ROOT, 'temp', 'repeat_render'))
    parser.add_argument('--child')
    parser.add_argument('--setup', default='default')
    parser.add_argument('--setups', default=','.join(SETUPS))
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if args.child:
        return child(args.child, args.setup, out)

    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from unleashed import core
    core.pre_check()
    RR.make_inputs(out)
    env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE='1', PYTHONUNBUFFERED='1')
    for setup in args.setups.split(','):
        labels = [f'trace_{setup}_1', f'trace_{setup}_2']
        ok = True
        for label in labels:
            t0 = time.time()
            with open(os.path.join(out, label + '.log'), 'w') as log:
                code = subprocess.call([sys.executable, os.path.abspath(__file__), '--child', label, '--setup', setup,
                                        '--out', out], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            ok = ok and code == 0
            print(f'{label}: {"done" if code == 0 else f"FAILED (exit {code}, see {label}.log)"} in {time.time() - t0:.0f} s', flush=True)
        if ok:
            print(f'{setup}: run 1 vs run 2', flush=True)
            same = compare(labels[0], labels[1], out)
            print(f'RESULT {setup}: {"identical" if same else "differ"}', flush=True)


if __name__ == '__main__':
    main()
