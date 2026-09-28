"""Do the models return a wrong output when several threads use them at once? (GPU)

    python tools/ort_race.py            (stop the Unleashed server first)

render_trace found one frame whose swap slice came out different (max 170
levels) for exactly the same input. This loads the swap model (inswapper)
and the mask model (DFL XSeg) the way the app does, makes 16 fixed inputs
for each, takes their outputs one at a time as the reference, then lets 8
threads call both models 200 times each, per way of calling:

    old      io_binding, output left on the GPU, copy_outputs_to_cpu()
             (the app until 48015a6; L4: 1 of 1600 swap calls wrong, off
             by 0.97; the others 0 -- the app now binds outputs to the CPU)
    sync     the same + io_binding.synchronize_outputs() before the copy
    cpu_out  io_binding, output bound to the CPU
    run      session.run()

and counts the calls whose output is not the reference. Writes nothing.
"""
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THREADS, CALLS = 8, 200


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import unleashed.globals as G
    from settings import Settings
    G.CFG = Settings('config.yaml')                          # read only
    from unleashed import core
    core.pre_check()
    G.execution_providers = core.decode_execution_providers([G.CFG.provider])
    from unleashed.utilities import get_device
    from unleashed.processors.FaceSwapInsightFace import FaceSwapInsightFace
    from unleashed.processors.Mask_XSeg import Mask_XSeg
    device = get_device()
    swap = FaceSwapInsightFace()
    swap.Initialize({'devicename': device, 'modelname': 'inswapper_128.onnx'})
    mask = Mask_XSeg()
    mask.Initialize({'devicename': device})
    print(f'device {device}, providers {G.execution_providers}', flush=True)

    rng = np.random.default_rng(0)
    models = {
        'swap': (swap.model_swap_insightface, 'output',
                 [{'target': rng.random((1, 3, 128, 128), dtype=np.float32),
                   'source': (lambda v: (v / np.linalg.norm(v)).astype(np.float32))(rng.standard_normal((1, 512)))}
                  for _ in range(16)]),
        'mask': (mask.model_xseg, mask.model_outputs[0].name,
                 [{mask.model_inputs[0].name: rng.random((1, 256, 256, 3), dtype=np.float32)} for _ in range(16)]),
    }

    def call(way, session, out_name, feeds):
        if way == 'run':
            return session.run([out_name], feeds)[0]
        binding = session.io_binding()
        for name, value in feeds.items():
            binding.bind_cpu_input(name, value)
        binding.bind_output(out_name, 'cpu' if way == 'cpu_out' else device)
        session.run_with_iobinding(binding)
        if way == 'sync':
            binding.synchronize_outputs()
        return binding.copy_outputs_to_cpu()[0]

    reference = {m: [call('run', s, o, f) for f in feeds] for m, (s, o, feeds) in models.items()}
    for way in ('old', 'sync', 'cpu_out', 'run'):
        same = all(np.array_equal(call(way, s, o, f), reference[m][i])
                   for m, (s, o, feeds) in models.items() for i, f in enumerate(feeds))
        wrong = {m: 0 for m in models}
        worst = {m: 0.0 for m in models}
        lock = threading.Lock()

        def worker(t):
            for c in range(CALLS):
                i = (t * 7 + c) % 16
                for m, (s, o, feeds) in models.items():
                    r = call(way, s, o, feeds[i])
                    if not np.array_equal(r, reference[m][i]):
                        d = float(np.abs(r.astype(np.float64) - reference[m][i]).max())
                        with lock:
                            wrong[m] += 1
                            worst[m] = max(worst[m], d)

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(f'{way:8s}: one at a time = reference: {same}; {THREADS} threads x {CALLS} calls: '
              + ', '.join(f'{m} {wrong[m]} wrong (max diff {worst[m]:.3g})' for m in models)
              + f'  [{time.time() - t0:.1f} s]', flush=True)


if __name__ == '__main__':
    main()
