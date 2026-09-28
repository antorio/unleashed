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


class OneCallAtATime(_BASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        on_gpu = any(p != 'CPUExecutionProvider' for p in self.get_providers())
        self._one_call = threading.Lock() if on_gpu else None

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
