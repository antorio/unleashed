"""Which setup stops single frames from coming out different? (GPU, real renders)

Colab cell (stop the Unleashed server first):
    %cd /content/unleashed
    !git pull -q
    !python tools/race_hunt.py

ort_race could not make the models go wrong on their own (0 of 4800 calls
per model, whichever way they were called), yet renders now and then still
give one frame far off (100+ levels), mostly one of the first frames. This
renders repeat_render's clip for real (ER off), 4 times inside each process
-- the first render starts the models cold, the other three reuse them
warm -- with three processes at once, per setup:

    app          the app as it is
    locked       every ONNX session runs one call at a time
    no_search    cuDNN's DEFAULT algorithm and no max workspace, instead of
                 the EXHAUSTIVE search on the first calls

A frame's right picture is the one most renders of the setup agree on. For
each setup it counts the frames that differ from it, cold and warm renders
apart, and lists them. Writes only to --out (/content/unleashed_test).
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
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('repeat_render', os.path.join(HERE, 'repeat_render.py'))
RR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RR)
ROOT = RR.ROOT
SETUPS = ['app', 'locked', 'no_search']
PROCESSES, RENDERS = 3, 4


def child(label, setup, renders, out):
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    work = os.path.join(out, label)
    os.makedirs(work, exist_ok=True)
    if setup == 'locked':                    # before insightface subclasses InferenceSession
        import onnxruntime as ort
        base = ort.InferenceSession

        class OneAtATime(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._one = threading.Lock()

            def run(self, *args, **kwargs):
                with self._one:
                    return super().run(*args, **kwargs)

            def run_with_iobinding(self, *args, **kwargs):
                with self._one:
                    return super().run_with_iobinding(*args, **kwargs)
        ort.InferenceSession = OneAtATime
    import unleashed.globals as G
    if setup == 'no_search':
        G.cudnn_conv_algo_search = 'DEFAULT'
        G.cudnn_conv_use_max_workspace = False
    from settings import Settings
    G.CFG = Settings('config.yaml')                          # never saved back
    G.CFG.config_file = os.path.join(work, 'config_copy.yaml')
    G.CFG.output_folder = G.output_path = work
    from unleashed import core
    import unleashed.ProcessMgr as PM
    from unleashed.ProcessEntry import ProcessEntry
    from ui.tabs import faceswap_state as S
    G.execution_providers = core.decode_execution_providers([G.CFG.provider])
    G.video_encoder, G.video_quality = G.CFG.output_video_codec, G.CFG.video_quality

    current = {'render': 0}
    frames = {}
    original = PM.ProcessMgr.process_frame

    def keep(self, frame, frame_index=None):
        result = original(self, frame, frame_index)
        if result is not None:
            frames[f"{current['render']}/{frame_index}"] = result.copy()
        return result
    PM.ProcessMgr.process_frame = keep

    clip, source = RR.make_inputs(out)
    S.add_sources([source])
    S.values.update(S.FACTORY)
    S.values.update({'mode': 'All faces', 'er': False})
    S.apply_settings()
    v = dict(S.values)
    plugin, _ = S.mask_plugin()
    for r in range(renders):
        current['render'] = r
        core.batch_process_regular(S.SWAP_MODEL, 'File', [ProcessEntry(clip, 0, 0, 0)], plugin, v['mask_objects'],
                                   True, None, bool(v['keep_mouth']), bool(v['keep_eyes']), int(v['passes']), None,
                                   S.active_source_index())
    np.savez_compressed(os.path.join(out, label + '.npz'), **frames)
    print(f'{label}: {len(frames)} frames', flush=True)
    os._exit(0)


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def report(setup, labels, out):
    data = {label: np.load(os.path.join(out, label + '.npz')) for label in labels}
    by_frame = {}
    for label, d in data.items():
        for key in d.files:
            r, i = (int(x) for x in key.split('/'))
            by_frame.setdefault(i, []).append((label, r, md5(d[key])))
    off = {'cold': [], 'warm': []}
    total = {'cold': 0, 'warm': 0}
    for i, rows in sorted(by_frame.items()):
        right, votes = Counter(h for _, _, h in rows).most_common(1)[0]
        ref = next(data[l][f'{r}/{i}'] for l, r, h in rows if h == right)
        for label, r, h in rows:
            kind = 'cold' if r == 0 else 'warm'
            total[kind] += 1
            if h != right:
                d = np.abs(data[label][f'{r}/{i}'].astype(np.int16) - ref.astype(np.int16)).max(axis=2)
                off[kind].append(f'{label} render {r + 1} frame {i} (max {int(d.max())}, {int((d > 2).sum())} px >2)')
    print(f'{setup}: cold renders {len(off["cold"])}/{total["cold"]} frames off, warm renders {len(off["warm"])}/{total["warm"]} frames off', flush=True)
    for line in (off['cold'] + off['warm'])[:12]:
        print(f'    {line}', flush=True)
    return len(off['cold']) + len(off['warm'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='/content/unleashed_test' if os.path.isdir('/content') else os.path.join(ROOT, 'temp', 'repeat_render'))
    parser.add_argument('--setups', default=','.join(SETUPS))
    parser.add_argument('--child')
    parser.add_argument('--setup', default='app')
    parser.add_argument('--renders', type=int, default=RENDERS)
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if args.child:
        return child(args.child, args.setup, args.renders, out)

    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from unleashed import core
    core.pre_check()
    RR.make_inputs(out)
    env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE='1', PYTHONUNBUFFERED='1')
    summary = {}
    for setup in args.setups.split(','):
        labels = [f'hunt_{setup}_{p + 1}' for p in range(PROCESSES)]
        t0 = time.time()
        procs = []
        for label in labels:
            log = open(os.path.join(out, label + '.log'), 'w')
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), '--child', label, '--setup', setup,
                                           '--renders', str(args.renders), '--out', out],
                                          stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT))
        codes = [p.wait() for p in procs]
        done = [l for l, c in zip(labels, codes) if c == 0]
        print(f'{setup}: {PROCESSES} processes x {args.renders} renders in {time.time() - t0:.0f} s'
              + ('' if len(done) == len(labels) else f' ({len(labels) - len(done)} FAILED, see their .log)'), flush=True)
        if done:
            summary[setup] = report(setup, done, out)
    print('RESULT: ' + ', '.join(f'{s} {n} frames off' for s, n in summary.items()), flush=True)


if __name__ == '__main__':
    main()
