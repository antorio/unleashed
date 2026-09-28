"""Does a render come out the same every time?

    python tools/repeat_render.py            (stop the Unleashed server first)

Makes a short neutral clip from insightface's sample photo t1.jpg: the three
people on its left, the picture turning 0 -> 12 degrees and drifting, the
last frames blurred. The source face is one of the people on the right.
The clip is rendered with the app's own Start path: in memory, factory
settings, All faces, the config's threads. Each run is its own process:

    ER off twice, ER on three times one after another, ER on three at once.

Every frame is compared exactly with the first run of the same kind. Seen on
a CPU Mac (28 Sep): with ER on, 2 of ~12 renders made while other renders ran
differed from frame 2 on, in every face; without that load they were always
identical. This checks the GPU. Writes only to --out (/content/unleashed_test
on Colab), never to the output folder or config.yaml.
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = 24
SEQUENTIAL = [('off1', 0), ('off2', 0), ('on1', 1), ('on2', 1), ('on3', 1)]
AT_ONCE = [('together1', 1), ('together2', 1), ('together3', 1)]


def sample_photo():
    import insightface
    return os.path.join(os.path.dirname(insightface.__file__), 'data', 'images', 't1.jpg')


def make_inputs(out):
    """The clip (lossless) and the source photo, made once."""
    clip, source = os.path.join(out, 'clip.mkv'), os.path.join(out, 'source.jpg')
    if os.path.exists(clip) and os.path.exists(source):
        return clip, source
    photo = cv2.imread(sample_photo())
    half = photo.shape[1] // 2
    cv2.imwrite(source, photo[:, half:])
    left = photo[:, :half]
    s = 640.0 / left.shape[1]
    left = cv2.resize(left, (int(left.shape[1] * s) // 2 * 2, int(left.shape[0] * s) // 2 * 2))
    h, w = left.shape[:2]
    writer = cv2.VideoWriter(clip + '.part.mkv', cv2.VideoWriter_fourcc(*'FFV1'), 10, (w, h))
    for i in range(FRAMES):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0.5 * i, 1.0 + 0.002 * i)
        m[:, 2] += (0.8 * i, 0.4 * i)
        frame = cv2.warpAffine(left, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if i >= FRAMES - 6:
            frame = cv2.GaussianBlur(frame, (0, 0), 0.8 * (i - FRAMES + 7))
        writer.write(frame)
    writer.release()
    os.replace(clip + '.part.mkv', clip)
    return clip, source


def child(label, er, out):
    """One render in this process; the frames go to <out>/<label>.npz."""
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    work = os.path.join(out, label)
    os.makedirs(work, exist_ok=True)
    import unleashed.globals as G
    from settings import Settings
    G.CFG = Settings('config.yaml')                          # the app's settings, never saved back
    G.CFG.config_file = os.path.join(work, 'config_copy.yaml')
    G.CFG.output_folder = G.output_path = work
    from unleashed import core
    import unleashed.ProcessMgr as PM
    from unleashed.ProcessEntry import ProcessEntry
    from ui.tabs import faceswap_state as S
    G.execution_providers = core.decode_execution_providers([G.CFG.provider])
    G.video_encoder, G.video_quality = G.CFG.output_video_codec, G.CFG.video_quality

    frames = {}
    original = PM.ProcessMgr.process_frame

    def keep(self, frame, frame_index=None):
        result = original(self, frame, frame_index)
        frames[frame_index] = None if result is None else result.copy()
        return result
    PM.ProcessMgr.process_frame = keep

    clip, source = make_inputs(out)
    print(S.add_sources([source]), flush=True)
    S.values.update(S.FACTORY)
    S.values.update({'mode': 'All faces', 'er': bool(er)})
    S.apply_settings()
    v = dict(S.values)
    plugin, _ = S.mask_plugin()
    core.batch_process_regular(S.SWAP_MODEL, 'File', [ProcessEntry(clip, 0, 0, 0)], plugin, v['mask_objects'],
                               True, None, bool(v['keep_mouth']), bool(v['keep_eyes']), int(v['passes']), None,
                               S.active_source_index())
    got = {str(i): f for i, f in frames.items() if f is not None}
    np.savez_compressed(os.path.join(out, label + '.npz'), **got)
    print(f'{label}: {len(got)} frames, threads {G.execution_threads}, providers {G.execution_providers}', flush=True)
    os._exit(0)


def start(label, er, out):
    log = open(os.path.join(out, label + '.log'), 'w')
    env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE='1', PYTHONUNBUFFERED='1')
    return subprocess.Popen([sys.executable, os.path.abspath(__file__), '--child', label, '--er', str(er), '--out', out],
                            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)


def finished(label, proc, out, t0):
    code = proc.wait()
    ok = code == 0 and os.path.exists(os.path.join(out, label + '.npz'))
    print(f'  {label}: {"done" if ok else f"FAILED (exit {code}, see {label}.log)"} in {time.time() - t0:.0f} s', flush=True)
    return ok


def compare(label, reference, out):
    a, b = np.load(os.path.join(out, reference + '.npz')), np.load(os.path.join(out, label + '.npz'))
    keys = sorted(set(a.files) | set(b.files), key=int)
    differ = []
    for k in keys:
        if k not in a.files or k not in b.files:
            differ.append(f'{k} (missing)')
        elif not np.array_equal(a[k], b[k]):
            d = np.abs(a[k].astype(np.int16) - b[k].astype(np.int16)).max(axis=2)
            differ.append(f'{k} (max {int(d.max())}, {int((d > 2).sum())} px >2)')
    text = 'identical' if not differ else f'{len(differ)} frames differ: ' + ', '.join(differ[:6]) + (' ...' if len(differ) > 6 else '')
    print(f'  {label} vs {reference}: {len(keys)} frames, {text}', flush=True)
    return not differ


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='/content/unleashed_test' if os.path.isdir('/content') else os.path.join(ROOT, 'temp', 'repeat_render'))
    parser.add_argument('--child')
    parser.add_argument('--er', type=int, default=1)
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if args.child:
        return child(args.child, args.er, out)

    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from unleashed import core
    core.pre_check()                                          # the models, once, before any render
    make_inputs(out)
    print(f'clip: {FRAMES} frames, 3 faces; results in {out}', flush=True)
    ok = {}
    print('one after another:', flush=True)
    for label, er in SEQUENTIAL:
        t0 = time.time()
        ok[label] = finished(label, start(label, er, out), out, t0)
    print('three at once:', flush=True)
    t0 = time.time()
    procs = [(label, start(label, er, out)) for label, er in AT_ONCE]
    for label, proc in procs:
        ok[label] = finished(label, proc, out, t0)

    print('compared frame by frame:', flush=True)
    same = {}
    for label, er in SEQUENTIAL + AT_ONCE:
        reference = 'on1' if er else 'off1'
        if label != reference and ok.get(label) and ok.get(reference):
            same[label] = compare(label, reference, out)
    off = [l for l in same if l.startswith('off')]
    on = [l for l in same if not l.startswith('off')]
    print(f'RESULT: ER off {sum(not same[l] for l in off)} of {len(off)} repeat renders differ; '
          f'ER on {sum(not same[l] for l in on)} of {len(on)} differ '
          f'(one after another {sum(not same[l] for l in on if l.startswith("on"))}, '
          f'at once {sum(not same[l] for l in on if l.startswith("together"))})', flush=True)


if __name__ == '__main__':
    main()
