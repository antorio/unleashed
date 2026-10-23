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

A call on the GPU that fails with a CUDA / cuDNN error is run again (up to
unleashed.globals.onnx_gpu_retries times) before the error counts. With
onnxruntime 1.30 + cuDNN 9.2x, on the L4 and on a T4, a depthwise convolution
of a face analysis model (2d106det / genderage: conv_N_dw_conv2d) failed now
and then with CUDNN_STATUS_EXECUTION_FAILED ("cuLaunchKernel ... invalid
argument"), about once in 5-15 minutes of rendering, same input sizes as
always; the session kept working afterwards. The analysis of that frame
failed as a whole, so the frame counted as "no face" and was not swapped.
onnxruntime's own answer to a GPU error in run() (an EPFail) was to move the
model to the CPU for good; that now happens only when the retries fail too.
Every GPU error is printed ("[gpu] ...") and counted (gpu_events_since), with
what was being processed (working_on: "frame 4647", a file name).
"""
import contextlib
import os
import re
import threading

import onnxruntime
from onnxruntime.capi.onnxruntime_pybind11_state import EPFail

_BASE = onnxruntime.InferenceSession

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


# what the calling thread is processing, for the GPU error messages
where = threading.local()

_events = []                # GPU errors: dicts with model, where, error, recovered
_events_lock = threading.Lock()


@contextlib.contextmanager
def working_on(label):
    """GPU errors inside this block name `label` ('frame 4647', a file name)."""
    before = getattr(where, 'label', None)
    where.label = label
    try:
        yield
    finally:
        where.label = before


def gpu_event_count():
    with _events_lock:
        return len(_events)


def gpu_events_since(count):
    """The GPU errors recorded after gpu_event_count() returned `count`."""
    with _events_lock:
        return list(_events[count:])


def _is_gpu_error(error):
    text = str(error).lower()
    return isinstance(error, EPFail) or any(m in text for m in ('cudnn', 'cuda', 'cublas', 'culaunch'))


def _short(error):
    """The telling part of an onnxruntime error (they run on for lines)."""
    text = str(error)
    m = re.search(r'(CUDNN(?:_FE)? failure \d+: \w+|CUDA failure \d+: [^;\n]+|CUBLAS failure \d+: \w+|'
                  r'HEURISTIC_QUERY_FAILED|CUDNN_STATUS_\w+)', text)
    return m.group(1).strip() if m else text.strip().splitlines()[0][:120]


def _record(session, error, outcome, retries):
    """outcome: 'retry' (a retry worked), 'cpu' (moved to the CPU, the call
    worked there) or 'failed' (the error reaches the caller)."""
    model = os.path.basename(getattr(session, '_model_path', None) or 'a model')
    label = getattr(where, 'label', None)
    with _events_lock:
        _events.append({'model': model, 'where': label, 'error': _short(error), 'recovered': outcome != 'failed'})
    at = f', {label}' if label else ''
    tried = f'also on {retries} {"retry" if retries == 1 else "retries"}'
    print(f'[gpu] {model}{at}: GPU error ({_short(error)}), ' +
          {'retry': 'the retry worked',
           'cpu': f'{tried}: the model now runs on the CPU',
           'failed': f'{tried}: this call failed'}[outcome], flush=True)


class OneCallAtATime(_BASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        applied = self.get_providers()
        _warn_if_gpu_missing(kwargs.get('providers', args[2] if len(args) > 2 else None), applied)
        self._on_gpu = any(p != 'CPUExecutionProvider' for p in applied)
        self._one_call = threading.Lock() if self._on_gpu else None
        # onnxruntime's run() moves the session to the CPU on the first GPU
        # error (EPFail); _call() retries on the GPU first and falls back the
        # same way only when the retries fail too
        self._cpu_fallback = self._on_gpu and getattr(self, '_enable_fallback', False)
        if self._cpu_fallback:
            self.disable_fallback()

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

    def _call(self, fn, args, kwargs, may_fall_back):
        lock = self._lock()
        with lock if lock is not None else contextlib.nullcontext():
            if not self._on_gpu:
                return fn(*args, **kwargs)
            import unleashed.globals
            retries = max(0, int(getattr(unleashed.globals, 'onnx_gpu_retries', 2) or 0))
            first = None
            for attempt in range(retries + 1):
                try:
                    result = fn(*args, **kwargs)
                except Exception as e:
                    if not _is_gpu_error(e):
                        raise
                    if first is None:
                        first = e
                    if attempt < retries:
                        continue
                    if may_fall_back and self._cpu_fallback and isinstance(e, EPFail):
                        # what onnxruntime's run() does on an EPFail
                        print(f'EP Error: {e!s} using {self._providers}')
                        print(f'Falling back to {self._fallback_providers} and retrying.')
                        self._cpu_fallback = False
                        self.set_providers(self._fallback_providers)
                        self._on_gpu = any(p != 'CPUExecutionProvider' for p in self.get_providers())
                        result = fn(*args, **kwargs)
                        _record(self, first, 'cpu', retries)
                        return result
                    _record(self, first, 'failed', retries)
                    raise
                if first is not None:
                    _record(self, first, 'retry', attempt)
                return result

    def run(self, *args, **kwargs):
        return self._call(super().run, args, kwargs, True)

    def run_with_iobinding(self, *args, **kwargs):
        # no CPU fallback here, in onnxruntime either (the binding belongs to
        # the session as it is)
        return self._call(super().run_with_iobinding, args, kwargs, False)


def install():
    if onnxruntime.InferenceSession is _BASE:
        onnxruntime.InferenceSession = OneCallAtATime
