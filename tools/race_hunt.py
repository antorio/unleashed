"""Do single frames still come out different, with and without the ONNX guard? (GPU, real renders)

Colab cell (stop the Unleashed server first):
    %cd /content/unleashed
    !git pull -q
    !python tools/race_hunt.py

Since model outputs go straight to host memory, renders gave 2 odd frames
in ~27 (before: 5 in 13); a first race_hunt (3 processes x 4 renders per
setup) found 0 for the app and 0 with every session locked, cuDNN DEFAULT
was 2.5x slower. The app now runs every ONNX session one call at a time
(unleashed/onnx_guard.py). This renders repeat_render's clip for real
(ER off), --rounds of --processes processes at once, --renders per process
(the first one starts the models cold, the others reuse them warm), per
setup:

    app          the app as it is (the guard on)
    unlocked     the guard off (unleashed.globals.onnx_one_call_per_session)

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
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('repeat_render', os.path.join(HERE, 'repeat_render.py'))
RR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RR)
ROOT = RR.ROOT
SETUPS = ['app', 'unlocked']
PROCESSES, RENDERS, ROUNDS = 3, 2, 4


def child(label, setup, renders, out):
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    work = os.path.join(out, label)
    os.makedirs(work, exist_ok=True)
    import unleashed.globals as G
    if setup == 'unlocked':
        G.onnx_one_call_per_session = False
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
    majority = {i: Counter(h for _, _, h in rows).most_common(1)[0][0] for i, rows in by_frame.items()}
    return len(off['cold']) + len(off['warm']), majority


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='/content/unleashed_test' if os.path.isdir('/content') else os.path.join(ROOT, 'temp', 'repeat_render'))
    parser.add_argument('--setups', default=','.join(SETUPS))
    parser.add_argument('--child')
    parser.add_argument('--setup', default='app')
    parser.add_argument('--renders', type=int, default=RENDERS, help='renders per process')
    parser.add_argument('--processes', type=int, default=PROCESSES, help='processes at once')
    parser.add_argument('--rounds', type=int, default=ROUNDS)
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
        t0 = time.time()
        done, failed = [], 0
        for rnd in range(args.rounds):
            labels = [f'hunt_{setup}_{rnd + 1}_{p + 1}' for p in range(args.processes)]
            procs = []
            for label in labels:
                log = open(os.path.join(out, label + '.log'), 'w')
                procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), '--child', label, '--setup', setup,
                                               '--renders', str(args.renders), '--out', out],
                                              stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT))
            codes = [p.wait() for p in procs]
            done += [l for l, c in zip(labels, codes) if c == 0]
            failed += sum(c != 0 for c in codes)
        print(f'{setup}: {args.rounds} rounds x {args.processes} processes x {args.renders} renders in {time.time() - t0:.0f} s'
              + ('' if not failed else f' ({failed} FAILED, see their .log)'), flush=True)
        if done:
            summary[setup] = report(setup, done, out)
    if len(summary) > 1:                     # does the guard change the picture?
        names = list(summary)
        first = summary[names[0]][1]
        for other in names[1:]:
            theirs = summary[other][1]
            same = sum(first.get(i) == h for i, h in theirs.items())
            print(f'{names[0]} and {other} agree on the picture of {same} of {len(theirs)} frames', flush=True)
    print('RESULT: ' + ', '.join(f'{s} {n} frames off' for s, (n, _) in summary.items()), flush=True)


if __name__ == '__main__':
    main()
