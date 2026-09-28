# every ONNX Runtime session runs one call at a time (see onnx_guard); this
# has to happen before insightface is imported, and all of unleashed's
# imports of it come after this package loads
from unleashed.onnx_guard import install as _install_onnx_guard

_install_onnx_guard()
