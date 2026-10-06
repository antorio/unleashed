"""One call at a time per ONNX Runtime GPU session.

The render's worker threads (8 on Colab) share every model session: the
face analyser's, the swap model's, the masks', the enhancers'. On the L4 a
swap slice came back wrong now and then for exactly the same input
(tools/render_trace.py), and earlier 8 threads on one LivePortrait session
crashed cuDNN (hence its own _LP_LOCK). Letting each GPU session run one
call at a time was not slower where it was measured (swap 128 / detector /
recognition from 8 threads in one process: tools/ort_race.py 130 vs
138-144 s; real renders, 3 processes: tools/race_hunt.py 105 vs 107 s);
different sessions still run side by side. Enhancers and 512 px were not
measured. Sessions on the CPU are left alone: there the lock cost 10-20% of
throughput with 8 threads and no wrong outputs were ever seen.

install() swaps onnxruntime.InferenceSession for this subclass. It must run
before insightface is imported (its model zoo subclasses InferenceSession
when it loads), which unleashed/__init__.py makes sure of.
unleashed.globals.onnx_one_call_per_session = False turns the lock off (A/B).
"""
import threading

import onnxruntime

_BASE = onnxruntime.InferenceSession


import os

_warned = set()
_BAR = '!' * 78


def _banner(kind, text):
    """Print a loud warning once per kind."""
    if kind in _warned:
        return
    _warned.add(kind)
    print(f'\n{_BAR}\n{text}\n{_BAR}\n', flush=True)


def _warn_if_gpu_missing(requested, applied):
    """ONNX Runtime falls back to the CPU quietly when the GPU provider cannot
    load (onnxruntime-gpu 1.21 on Colab after it moved to CUDA 13:
    libcublasLt.so.12 missing); a render then takes hours. Say so once,
    loudly."""
    names = [p[0] if isinstance(p, (tuple, list)) else p for p in requested or []]
    available = onnxruntime.get_available_providers()     # a CPU-only build (Mac) lists no CUDA
    wanted = [n for n in names if n in available and n not in ('CPUExecutionProvider', 'TensorrtExecutionProvider')]
    if not wanted or any(n in applied for n in wanted):
        return
    _banner('load', f'!!! GPU NOT USED: {", ".join(wanted)} could not load, every model runs on the CPU.\n'
                    f'!!! See the error just above (a missing CUDA library?); check with: python tools/gpu_check.py')


class OneCallAtATime(_BASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        applied = self.get_providers()
        _warn_if_gpu_missing(kwargs.get('providers', args[2] if len(args) > 2 else None), applied)
        on_gpu = any(p != 'CPUExecutionProvider' for p in applied)
        self._one_call = threading.Lock() if on_gpu else None

    def set_providers(self, *args, **kwargs):
        """run() calls this when the GPU fails mid-way (onnxruntime's fallback:
        "EP Error ... Falling back to ['CPUExecutionProvider'] and retrying");
        the model then quietly stays on the CPU. Seen on a T4 with cuDNN 9.27:
        the face recognition model moved to the CPU, no banner."""
        super().set_providers(*args, **kwargs)
        if self._one_call is not None and all(p == 'CPUExecutionProvider' for p in self.get_providers()):
            model = os.path.basename(getattr(self, '_model_path', None) or 'a model')
            _banner('switched', f'!!! GPU NOT USED for {model}: it hit a GPU error and now runs on the CPU.\n'
                                f'!!! See the "EP Error" just above; check with: python tools/gpu_check.py')

    def _lock(self):
        import unleashed.globals
        if self._one_call is None or not getattr(unleashed.globals, 'onnx_one_call_per_session', True):
            return None
        return self._one_call

    def run(self, *args, **kwargs):
        lock = self._lock()
        if lock is None:
            return super().run(*args, **kwargs)
        with lock:
            return super().run(*args, **kwargs)

    def run_with_iobinding(self, *args, **kwargs):
        lock = self._lock()
        if lock is None:
            return super().run_with_iobinding(*args, **kwargs)
        with lock:
            return super().run_with_iobinding(*args, **kwargs)


def install():
    if onnxruntime.InferenceSession is _BASE:
        onnxruntime.InferenceSession = OneCallAtATime
