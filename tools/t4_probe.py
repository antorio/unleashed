"""Which CUDA setting lets the models run on a T4? (research, branch t4-research)

Colab cell (stop the Unleashed server first; the app must have run once so
the models are downloaded):
    %cd /content/unleashed
    !git fetch -q origin t4-research
    !git checkout -q t4-research
    !python tools/t4_probe.py

On a T4 (Turing, sm_75) with onnxruntime-gpu 1.30 and the cuDNN 9.27 that
Colab's torch brings (CUDA 13), every convolution failed: "No valid engine
configs ... smVersion 750", HEURISTIC_QUERY_FAILED; the face recognition
model fell back to the CPU, the swap model failed outright. The L4 runs the
same cuDNN without trouble. This runs the app's models (swap, the five
buffalo_l models) plus the two convolutions from the error, once per setup,
each setup in its own process (a failed convolution leaves the session
broken):

    app            the app's options: cuDNN search EXHAUSTIVE, max workspace, TF32 on
    tf32_off       app + use_tf32 0 (Turing has no TF32)
    search_default cuDNN search DEFAULT
    search_heur    cuDNN search HEURISTIC
    no_workspace   app without max workspace
    all_off        DEFAULT search, no max workspace, TF32 off
    cudnn_9.14     app options with cuDNN 9.14 (the version onnxruntime 1.30
                   is built with), installed apart in /content/cudnn_9.14

For every model: did the session stay on the GPU, does it run, are the
numbers right (vs the CPU, relative to the largest value), how long a run
takes. Writes only /content/cudnn_9.14 (pip --target, cuDNN only).
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
CUDNN_OLD = '9.14.0.64'
APP = {'cudnn_conv_algo_search': 'EXHAUSTIVE', 'cudnn_conv_use_max_workspace': '1'}
SETUPS = {
    'app': (APP, None),
    'tf32_off': (dict(APP, use_tf32='0'), None),
    'search_default': ({'cudnn_conv_algo_search': 'DEFAULT', 'cudnn_conv_use_max_workspace': '1'}, None),
    'search_heur': ({'cudnn_conv_algo_search': 'HEURISTIC', 'cudnn_conv_use_max_workspace': '1'}, None),
    'no_workspace': ({'cudnn_conv_algo_search': 'EXHAUSTIVE', 'cudnn_conv_use_max_workspace': '0'}, None),
    'all_off': ({'cudnn_conv_algo_search': 'DEFAULT', 'cudnn_conv_use_max_workspace': '0', 'use_tf32': '0'}, None),
    'cudnn_9.14': (APP, CUDNN_OLD),
}


def conv_model(cin, cout, size, pad):
    """One 3x3 convolution, as in the error messages."""
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(cin + cout)
    w = numpy_helper.from_array((rng.standard_normal((cout, cin, 3, 3)) / np.sqrt(cin * 9)).astype(np.float32), 'w')
    out = size + 2 * pad - 2
    graph = helper.make_graph([helper.make_node('Conv', ['x', 'w'], ['y'], pads=[pad] * 4)], 'conv',
                              [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, cin, size, size])],
                              [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, cout, out, out])], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    return model.SerializeToString()


def model_list():
    items = [('conv 64->128 @56 (recognition Conv_19)', conv_model(64, 128, 56, 1)),
             ('conv 1024->1024 @34 (swap Conv_62)', conv_model(1024, 1024, 34, 0))]
    for name in ['inswapper_128.onnx', 'buffalo_l/w600k_r50.onnx', 'buffalo_l/det_10g.onnx',
                 'buffalo_l/1k3d68.onnx', 'buffalo_l/2d106det.onnx', 'buffalo_l/genderage.onnx']:
        path = os.path.join(MODELS, name)
        items.append((name, path if os.path.isfile(path) else None))
    return items


def feeds_for(session, rng):
    """Random inputs; an open batch size is 1, open image sizes 640 (detector)."""
    feeds = {}
    for i in session.get_inputs():
        shape = [d if isinstance(d, int) else (1 if n == 0 else 640) for n, d in enumerate(i.shape)]
        if i.name == 'source':                        # inswapper: a unit embedding
            v = rng.standard_normal(shape).astype(np.float32)
            feeds[i.name] = v / np.linalg.norm(v)
        else:
            feeds[i.name] = rng.random(shape, dtype=np.float32)
    return feeds


def install_old_cudnn(version):
    folder = f'/content/cudnn_{version.rsplit(".", 2)[0]}'
    lib = os.path.join(folder, 'nvidia', 'cudnn', 'lib')
    if not os.path.isfile(os.path.join(lib, 'libcudnn.so.9')):
        print(f'installing cuDNN {version} (CUDA 13) into {folder} ...', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--target', folder,
                        f'nvidia-cudnn-cu13=={version}'], check=True)
    return lib


def child(setup):
    options, cudnn = SETUPS[setup]
    import onnxruntime as ort
    if cudnn:
        lib = install_old_cudnn(cudnn)
        ctypes.CDLL(os.path.join(lib, 'libcudnn.so.9'))      # before anything else loads cuDNN
        ort.preload_dlls(cudnn=False)
    else:
        ort.preload_dlls()                                    # as unleashed/__init__.py
    try:
        import torch                                          # the app imports torch next
        gpu_name, torch_version = torch.cuda.get_device_name(0), torch.__version__
    except Exception:
        gpu_name, torch_version = 'none', 'none'
    results = []
    for name, src in model_list():
        if src is None:
            results.append({'model': name, 'status': 'missing'})
            continue
        row = {'model': name}
        try:
            gpu = ort.InferenceSession(src, providers=[('CUDAExecutionProvider', dict(options)), 'CPUExecutionProvider'])
            row['on_gpu'] = 'CUDAExecutionProvider' in gpu.get_providers()
            cpu = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
            feeds = feeds_for(cpu, np.random.default_rng(0))
            want = cpu.run(None, feeds)
            got = gpu.run(None, feeds)                        # first run: cuDNN search
            t0 = time.perf_counter()
            got = gpu.run(None, feeds)
            row['ms'] = round((time.perf_counter() - t0) * 1000, 1)
            row['rel_err'] = max(float(np.abs(g - w).max()) / max(float(np.abs(w).max()), 1e-6) for g, w in zip(got, want))
            row['status'] = 'ok' if row['on_gpu'] and row['rel_err'] < 1e-2 else ('cpu' if not row['on_gpu'] else 'wrong')
        except Exception as e:
            msg = str(e)
            row['status'] = 'error'
            row['error'] = 'HEURISTIC_QUERY_FAILED' if 'HEURISTIC_QUERY_FAILED' in msg else msg.splitlines()[0][:160]
        results.append(row)
    print('RESULT_JSON ' + json.dumps({'setup': setup, 'torch': torch_version, 'ort': ort.__version__,
                                       'gpu': gpu_name, 'rows': results}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--child')
    parser.add_argument('--setups', default=','.join(SETUPS))
    args = parser.parse_args()
    if args.child:
        return child(args.child)
    os.chdir(ROOT)
    summary = []
    for setup in args.setups.split(','):
        env = dict(os.environ)
        if SETUPS[setup][1]:                                  # cuDNN loads its sub-libraries by name too
            lib = install_old_cudnn(SETUPS[setup][1])
            env['LD_LIBRARY_PATH'] = lib + os.pathsep + env.get('LD_LIBRARY_PATH', '')
        t0 = time.time()
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', setup],
                              capture_output=True, text=True, env=env, cwd=ROOT)
        line = next((l for l in proc.stdout.splitlines() if l.startswith('RESULT_JSON ')), None)
        if line is None:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
            print(f'== {setup}: CRASHED (exit {proc.returncode}): ' + ' | '.join(tail), flush=True)
            summary.append((setup, 'crashed'))
            continue
        data = json.loads(line[len('RESULT_JSON '):])
        if not summary:
            print(f"GPU {data['gpu']}, torch {data['torch']}, onnxruntime {data['ort']}", flush=True)
        print(f'== {setup} ({time.time() - t0:.0f} s)', flush=True)
        for r in data['rows']:
            extra = (f" {r['ms']} ms, rel. diff {r['rel_err']:.1e}" if 'ms' in r else '') + \
                    (f" -- {r['error']}" if 'error' in r else '')
            print(f"    {r['status']:7s} {r['model']}{extra}", flush=True)
        good = sum(r['status'] == 'ok' for r in data['rows'])
        known = sum(r['status'] != 'missing' for r in data['rows'])
        summary.append((setup, f'{good}/{known} ok'))
    print('RESULT: ' + ', '.join(f'{s} {v}' for s, v in summary), flush=True)


if __name__ == '__main__':
    main()
