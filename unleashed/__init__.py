# ONNX Runtime's CUDA build loads its CUDA / cuDNN libraries from the NVIDIA
# wheels torch brings along (Colab: CUDA 13); preload_dlls() finds them there.
# No-op on CPU builds (Mac).
import onnxruntime as _ort

if hasattr(_ort, 'preload_dlls'):
    _ort.preload_dlls()

# every ONNX Runtime session runs one call at a time (see onnx_guard); this
# has to happen before insightface is imported, and all of unleashed's
# imports of it come after this package loads
from unleashed.onnx_guard import install as _install_onnx_guard

_install_onnx_guard()
