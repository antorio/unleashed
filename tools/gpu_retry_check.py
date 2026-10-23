"""Do GPU errors get retried, and does the render stay the same? (GPU, real renders)

Colab cell (stop the Unleashed server first):
    %cd /content/unleashed
    !git pull -q
    !python tools/gpu_retry_check.py

Renders repeat_render's clip (24 frames, 3 faces, 128 px) once per scenario,
each in its own process, with CUDA / cuDNN errors injected into the model
calls (the same message as the real L4 / T4 errors), and checks:

    base_old == base          old code (--old, default 356c42a) vs this one, no errors
    gender_forced == base     the gender/age model run anyway: nothing else changes
    transient == base         one error now and then in every model, 8 threads:
                              each retried, every error recovered
    persistent                every analysis call of frame 10 fails: reported as
                              "frame 10" (and only frames from 10 on change)
    epfail                    the recognition model always fails with an EPFail:
                              it moves to the CPU after the retries, render finishes
    swap_dead                 the swap model always fails: the render stops with
                              the error (no hang)
    women_old == women        All women: gender still read, same result
    er_base == er_transient   Expression Restorer on, errors in its models too

Writes only to --out (/content/unleashed_test). Without a GPU (a Mac) the
sessions are marked as GPU sessions so the retry path runs on the CPU.
"""
import argparse
import collections
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUDNN_MSG = ("Non-zero status code returned while running Conv node. Name:'conv_7_dw_conv2d' Status Message: "
             "CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED ; GPU=0 ; file=conv.cc ; line=494")
# name: (code, mode, Expression Restorer, injection)
SCENARIOS = {
    'base_old': ('old', 'All faces', False, None),
    'base': ('new', 'All faces', False, None),
    'gender_forced': ('new', 'All faces', False, 'force_gender'),
    'transient': ('new', 'All faces', False, 'transient'),
    'persistent': ('new', 'All faces', False, 'persistent_frame10'),
    'epfail': ('new', 'All faces', False, 'epfail_recognition'),
    'swap_dead': ('new', 'All faces', False, 'swap_dead'),
    'women_old': ('old', 'All women', False, None),
    'women': ('new', 'All women', False, None),
    'er_base': ('new', 'All faces', True, None),
    'er_transient': ('new', 'All faces', True, 'transient'),
}
TRANSIENT_CALLS = (2, 11, 29, 47)       # these calls of every model fail once


def _cuda_works(onnxruntime):
    """A session really runs on CUDA (a GPU build lists CUDA without a GPU)."""
    if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        return False
    from onnx import TensorProto, helper
    graph = helper.make_graph([helper.make_node('Relu', ['x'], ['y'])], 'probe',
                              [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    try:
        session = onnxruntime.InferenceSession(model.SerializeToString(), providers=['CUDAExecutionProvider'])
        return 'CUDAExecutionProvider' in session.get_providers()
    except Exception:
        return False


def child(name, root, out):
    code, mode, er, inj = SCENARIOS[name]
    os.chdir(root)
    sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location('repeat_render', os.path.join(HERE, 'repeat_render.py'))
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    work = os.path.join(out, 'work_' + name)
    os.makedirs(work, exist_ok=True)
    import unleashed.globals as G
    from settings import Settings
    G.CFG = Settings(os.path.join(ROOT, 'config.yaml'))     # the same settings for both code versions, never saved
    G.CFG.config_file = os.path.join(work, 'config_copy.yaml')
    G.CFG.output_folder = G.output_path = work
    import onnxruntime
    from onnxruntime.capi import onnxruntime_inference_collection as OIC
    from onnxruntime.capi.onnxruntime_pybind11_state import EPFail, Fail
    gpu = _cuda_works(onnxruntime)
    calls, injected = collections.Counter(), collections.Counter()
    lock = threading.Lock()
    guard = None
    if code == 'new':
        from unleashed import onnx_guard as guard
        if not gpu:                     # a Mac: let the retry path run on the CPU
            init = guard.OneCallAtATime.__init__

            def as_gpu(self, *a, **k):
                init(self, *a, **k)
                self._on_gpu, self._one_call, self._cpu_fallback = True, threading.Lock(), True
                self.disable_fallback()
            guard.OneCallAtATime.__init__ = as_gpu
    run, run_iob = OIC.Session.run, OIC.Session.run_with_iobinding

    def error_for(self, iobinding):
        model = os.path.basename(getattr(self, '_model_path', '') or '')
        with lock:
            calls[model] += 1
            n = calls[model]
        if not getattr(self, '_on_gpu', False):
            return None
        err = None
        if inj == 'transient' and n in TRANSIENT_CALLS:
            err = RuntimeError('Error in execution: ' + CUDNN_MSG) if iobinding else Fail(CUDNN_MSG)
        elif inj == 'persistent_frame10' and model == '2d106det.onnx' and getattr(guard.where, 'label', None) == 'frame 10':
            err = Fail(CUDNN_MSG)
        elif inj == 'epfail_recognition' and model == 'w600k_r50.onnx':
            err = EPFail('[ONNXRuntimeError] : 11 : EP_FAIL : ' + CUDNN_MSG)
        elif inj == 'swap_dead' and iobinding:
            err = RuntimeError('Error in execution: ' + CUDNN_MSG)
        if err is not None:
            with lock:
                injected[model] += 1
        return err

    def patched_run(self, *a, **k):
        err = error_for(self, False)
        if err is not None:
            raise err
        return run(self, *a, **k)

    def patched_iob(self, *a, **k):
        err = error_for(self, True)
        if err is not None:
            raise err
        return run_iob(self, *a, **k)
    OIC.Session.run, OIC.Session.run_with_iobinding = patched_run, patched_iob

    from unleashed import core
    import unleashed.ProcessMgr as PM
    from unleashed.ProcessEntry import ProcessEntry
    from ui.tabs import faceswap_state as S
    G.execution_providers = core.decode_execution_providers([G.CFG.provider])
    G.video_encoder, G.video_quality = G.CFG.output_video_codec, G.CFG.video_quality
    if inj == 'force_gender':
        initialize = PM.ProcessMgr.initialize

        def gender_always(self, *a, **k):
            initialize(self, *a, **k)
            G.face_gender_needed = True
        PM.ProcessMgr.initialize = gender_always
    frames = {}
    process_frame = PM.ProcessMgr.process_frame

    def keep(self, frame, frame_index=None):
        result = process_frame(self, frame, frame_index)
        if result is not None and frame_index is not None:
            frames[str(frame_index)] = result.copy()
        return result
    PM.ProcessMgr.process_frame = keep

    clip, source = rr.make_inputs(out)
    S.add_sources([source])
    S.values.update(S.FACTORY)
    S.values.update({'mode': mode, 'er': er, 'resolution': '128px'})
    S.apply_settings()
    v = dict(S.values)
    plugin, _ = S.mask_plugin()
    G.execution_threads = 8
    entry = ProcessEntry(clip, 0, 0, 0)
    before = guard.gpu_event_count() if guard else 0
    error = None
    t0 = time.time()
    try:
        core.batch_process_regular(S.SWAP_MODEL, 'File', [entry], plugin, v['mask_objects'], True, None,
                                   bool(v['keep_mouth']), bool(v['keep_eyes']), int(v['passes']), None,
                                   S.active_source_index())
    except Exception as e:
        error = f'{type(e).__name__}: {str(e).splitlines()[0][:120]}'
    np.savez_compressed(os.path.join(out, name + '.npz'), **frames)
    events = guard.gpu_events_since(before) if guard else []
    info = {'secs': round(time.time() - t0), 'error': error, 'frames': len(frames), 'gpu': gpu,
            'genderage_calls': calls.get('genderage.onnx', 0), 'injected': sum(injected.values()),
            'events': len(events), 'recovered': sum(e['recovered'] for e in events),
            'gpu_failed': getattr(entry, 'gpu_failed', None), 'gpu_recovered': getattr(entry, 'gpu_recovered', None),
            # the source photo is analysed before the render: errors there too
            'recovered_all': sum(e['recovered'] for e in guard.gpu_events_since(0)) if guard else 0}
    json.dump(info, open(os.path.join(out, name + '.json'), 'w'))
    print('INFO ' + json.dumps(info), flush=True)
    os._exit(0)


def frames_of(out, name):
    path = os.path.join(out, name + '.npz')
    return np.load(path) if os.path.exists(path) else None


def compare(out, a, b):
    """(identical, first differing frame number or None, text)"""
    A, B = frames_of(out, a), frames_of(out, b)
    if A is None or B is None:
        return False, None, 'missing'
    keys = sorted(set(A.files) | set(B.files), key=int)
    diff = [k for k in keys if k not in A.files or k not in B.files or not np.array_equal(A[k], B[k])]
    if not diff:
        return True, None, f'identical ({len(keys)} frames)'
    k = diff[0]
    d = np.abs(A[k].astype(np.int16) - B[k].astype(np.int16)).max() if k in A.files and k in B.files else -1
    return False, int(k) + 1, f'{len(diff)}/{len(keys)} frames differ, first frame {int(k) + 1} (max {d})'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='/content/unleashed_test' if os.path.isdir('/content') else os.path.join(ROOT, 'temp', 'gpu_retry_check'))
    parser.add_argument('--old', default='356c42a', help='commit with the code before the retries')
    parser.add_argument('--only', help='comma-separated scenarios')
    parser.add_argument('--child')
    parser.add_argument('--root')
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if args.child:
        return child(args.child, args.root, out)

    names = args.only.split(',') if args.only else list(SCENARIOS)
    old_root = os.path.join(out, 'old_code')
    if any(SCENARIOS[n][0] == 'old' for n in names):
        commit = subprocess.run(['git', 'rev-parse', args.old], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        subprocess.run(['git', 'worktree', 'remove', '--force', old_root], cwd=ROOT, capture_output=True)
        subprocess.run(['git', 'worktree', 'prune'], cwd=ROOT, capture_output=True)
        subprocess.run(['git', 'worktree', 'add', '-q', '--detach', old_root, commit], cwd=ROOT, check=True)
        os.symlink(os.path.join(ROOT, 'models'), os.path.join(old_root, 'models'))
        print(f'old code: {args.old} = {commit[:7]}', flush=True)
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from unleashed import core
    core.pre_check()
    env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE='1', PYTHONUNBUFFERED='1')
    infos = {}
    for name in names:
        root = old_root if SCENARIOS[name][0] == 'old' else ROOT
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', name, '--root', root, '--out', out],
                           capture_output=True, text=True, env=env, cwd=root)
        open(os.path.join(out, name + '.log'), 'w').write(p.stdout + p.stderr)
        line = next((l for l in p.stdout.splitlines() if l.startswith('INFO ')), None)
        if line is None:
            print(f'== {name}: CRASHED, see {name}.log: ' + ' | '.join((p.stdout + p.stderr).strip().splitlines()[-3:]), flush=True)
            continue
        infos[name] = json.loads(line[5:])
        i = infos[name]
        print(f"== {name} ({time.time() - t0:.0f} s): frames {i['frames']}, error {i['error']}, injected {i['injected']}, "
              f"GPU errors {i['events']} ({i['recovered']} recovered), not recovered at {i['gpu_failed']}, "
              f"gender/age runs {i['genderage_calls']}", flush=True)

    checks = []

    def check(label, ok, detail):
        checks.append(ok)
        print(f"{'OK  ' if ok else 'FAIL'} {label}: {detail}", flush=True)

    def same(a, b, label):
        if a in infos and b in infos:
            ok, _, text = compare(out, a, b)
            check(label, ok, text)
    print('checks:', flush=True)
    same('base_old', 'base', 'old vs new code, no errors')
    if 'base' in infos:
        check('gender/age not run in All faces', infos['base']['genderage_calls'] == 0, f"{infos['base']['genderage_calls']} runs")
    same('base', 'gender_forced', 'gender/age skipped vs run')
    if 'transient' in infos:
        i = infos['transient']
        same('base', 'transient', 'errors now and then, retried')
        check('every injected error recovered', i['events'] == i['recovered'] and i['injected'] >= i['events'] > 0,
              f"{i['injected']} injected, {i['events']} errors, {i['recovered']} recovered")
    if 'persistent' in infos:
        i = infos['persistent']
        _, first, text = compare(out, 'base', 'persistent')
        check('lasting error reported at frame 10', i['gpu_failed'] == ['frame 10'], f"reported {i['gpu_failed']}")
        check('frames before 10 unchanged', first == 10, text)
    if 'epfail' in infos:
        i = infos['epfail']
        check('EPFail: render finishes, model moved to the CPU', i['error'] is None and i['frames'] == 24 and i['recovered_all'] >= 1,
              f"error {i['error']}, frames {i['frames']}, recovered {i['recovered_all']} (moved to the CPU)")
        if 'base' in infos:
            print(f"     epfail vs base: {compare(out, 'base', 'epfail')[2]} (the CPU computes slightly differently)", flush=True)
    if 'swap_dead' in infos:
        i = infos['swap_dead']
        check('swap model always failing: the render stops with the error', bool(i['error']) and 'CUDNN' in i['error'],
              f"{i['error']}")
    same('women_old', 'women', 'All women, old vs new')
    if 'women' in infos:
        check('gender/age run in All women', infos['women']['genderage_calls'] > 0, f"{infos['women']['genderage_calls']} runs")
    if 'er_transient' in infos:
        i = infos['er_transient']
        same('er_base', 'er_transient', 'Expression Restorer, errors now and then')
        check('ER: every injected error recovered', i['events'] == i['recovered'] > 0,
              f"{i['injected']} injected, {i['events']} errors, {i['recovered']} recovered")
    if os.path.isdir(old_root):
        subprocess.run(['git', 'worktree', 'remove', '--force', old_root], cwd=ROOT, capture_output=True)
    print(f'RESULT: {sum(checks)}/{len(checks)} checks OK', flush=True)


if __name__ == '__main__':
    main()
