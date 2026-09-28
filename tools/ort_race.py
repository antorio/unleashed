"""Do the models return a wrong output when several threads use them at once? (GPU)

    python tools/ort_race.py [--calls 600]     (stop the Unleashed server first)

render_trace found a frame whose swap slice came out different (max 170
levels) for exactly the same input. A first run of this test (1600 calls)
found 1 wrong inswapper output with the app's old way of calling it
(io_binding, output left on the GPU, then copy_outputs_to_cpu(): off by
0.97) and 0 with the output bound to the CPU -- too few calls to tell, and
renders still differed after the app switched to CPU outputs.

This loads the swap model (inswapper, as the app does) and buffalo_l's face
detector and recognition model, makes 16 fixed inputs for each, takes their
outputs one at a time as the reference, then lets 8 threads call all three
models --calls times each, per way of calling:

    old      io_binding, output on the GPU, copy_outputs_to_cpu() (swap only)
    cpu_out  io_binding, output bound to the CPU (the app now, swap)
    run      session.run() (the app, detector / recognition)
    locked   session.run() with one lock per session: one call at a time

and counts the outputs that differ from the reference. Writes nothing.
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THREADS = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--calls', type=int, default=600, help='calls per thread and model, per way')
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import unleashed.globals as G
    from settings import Settings
    G.CFG = Settings('config.yaml')                          # read only
    from unleashed import core
    core.pre_check()
    G.execution_providers = core.decode_execution_providers([G.CFG.provider])
    from unleashed.utilities import get_device
    from unleashed.face_util import get_face_analyser
    from unleashed.processors.FaceSwapInsightFace import FaceSwapInsightFace
    device = get_device()
    swap = FaceSwapInsightFace()
    swap.Initialize({'devicename': device, 'modelname': 'inswapper_128.onnx'})
    analyser = get_face_analyser()
    print(f'device {device}, providers {G.execution_providers}', flush=True)

    rng = np.random.default_rng(0)
    unit = lambda v: (v / np.linalg.norm(v)).astype(np.float32)
    det = analyser.det_model.session
    rec = analyser.models['recognition'].session
    models = {
        'swap': (swap.model_swap_insightface,
                 [{'target': rng.random((1, 3, 128, 128), dtype=np.float32), 'source': unit(rng.standard_normal((1, 512)))}
                  for _ in range(16)]),
        'detect': (det, [{det.get_inputs()[0].name: rng.random((1, 3, 640, 640), dtype=np.float32)} for _ in range(16)]),
        'recognise': (rec, [{rec.get_inputs()[0].name: rng.standard_normal((1, 3, 112, 112)).astype(np.float32)} for _ in range(16)]),
    }
    locks = {m: threading.Lock() for m in models}

    def call(way, m, session, feeds):
        names = [o.name for o in session.get_outputs()]
        if way in ('old', 'cpu_out') and m == 'swap':
            binding = session.io_binding()
            for name, value in feeds.items():
                binding.bind_cpu_input(name, value)
            for name in names:
                binding.bind_output(name, device if way == 'old' else 'cpu')
            session.run_with_iobinding(binding)
            return binding.copy_outputs_to_cpu()
        if way == 'locked':
            with locks[m]:
                return session.run(names, feeds)
        return session.run(names, feeds)

    def equal(a, b):
        return all(np.array_equal(x, y) for x, y in zip(a, b))

    def diff(a, b):
        return max(float(np.abs(np.asarray(x, np.float64) - y).max()) for x, y in zip(a, b))

    reference = {m: [call('run', m, s, f) for f in feeds] for m, (s, feeds) in models.items()}
    for way in ('old', 'cpu_out', 'run', 'locked'):
        same = all(equal(call(way, m, s, f), reference[m][i]) for m, (s, feeds) in models.items() for i, f in enumerate(feeds))
        wrong = {m: 0 for m in models}
        worst = {m: 0.0 for m in models}
        lock = threading.Lock()

        def worker(t):
            for c in range(args.calls):
                i = (t * 7 + c) % 16
                for m, (s, feeds) in models.items():
                    r = call(way, m, s, feeds[i])
                    if not equal(r, reference[m][i]):
                        d = diff(r, reference[m][i])
                        with lock:
                            wrong[m] += 1
                            worst[m] = max(worst[m], d)

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        n = THREADS * args.calls
        print(f'{way:8s}: one at a time = reference: {same}; {THREADS} threads x {args.calls} calls: '
              + ', '.join(f'{m} {wrong[m]}/{n} wrong' + (f' (max diff {worst[m]:.3g})' if wrong[m] else '') for m in models)
              + f'  [{time.time() - t0:.0f} s]', flush=True)


if __name__ == '__main__':
    main()
