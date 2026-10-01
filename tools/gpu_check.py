"""Does ONNX Runtime really run on the GPU? (Colab: notebook cell after the install)

    python tools/gpu_check.py

onnxruntime.get_available_providers() lists CUDA even when its libraries
cannot load; the sessions then quietly run on the CPU (onnxruntime-gpu 1.21
on Colab after it moved to CUDA 13, Oct 2026). This starts like the app
(import unleashed, then torch), runs a small convolution + matrix model
(cuDNN and cuBLAS) on the CUDA provider, compares it with the CPU, and exits
with an error when the GPU is not used. Needs no model files.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny_model():
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.standard_normal((8, 3, 3, 3)).astype(np.float32), 'w')
    m = numpy_helper.from_array(rng.standard_normal((32, 16)).astype(np.float32), 'm')
    nodes = [helper.make_node('Conv', ['x', 'w'], ['c'], pads=[1, 1, 1, 1]),
             helper.make_node('MatMul', ['c', 'm'], ['y'])]
    graph = helper.make_graph(nodes, 'check', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 32, 32])],
                              [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 8, 32, 16])], [w, m])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    return model.SerializeToString()


def main():
    sys.path.insert(0, ROOT)
    import unleashed                                   # preloads the CUDA libraries, like the app
    import onnxruntime as ort
    try:
        import torch
        print(f'torch {torch.__version__} (CUDA {torch.version.cuda}), GPU: {torch.cuda.is_available()}')
    except ImportError:
        print('torch not installed')
    import cv2, gradio, insightface
    print(f'python {sys.version.split()[0]}, numpy {np.__version__}, insightface {insightface.__version__}, '
          f'cv2 {cv2.__version__}, gradio {gradio.__version__}')
    print(f'onnxruntime {ort.__version__}, providers listed: {ort.get_available_providers()}')

    model = tiny_model()
    x = np.random.default_rng(1).standard_normal((1, 3, 32, 32)).astype(np.float32)
    cpu = ort.InferenceSession(model, providers=['CPUExecutionProvider']).run(None, {'x': x})[0]
    gpu_session = ort.InferenceSession(model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    applied = gpu_session.get_providers()
    print(f'providers applied: {applied}')
    if 'CUDAExecutionProvider' not in applied:
        print('RESULT: FAILED, the GPU is NOT used (see the error above). Render would run on the CPU.')
        sys.exit(1)
    gpu = gpu_session.run(None, {'x': x})[0]
    # the CUDA provider uses TF32 for convolutions / matrix products by default
    # (about 3 significant digits), so compare relative to the output's size;
    # with TF32 off it has to match the CPU closely
    exact = ort.InferenceSession(model, providers=[('CUDAExecutionProvider', {'use_tf32': 0}), 'CPUExecutionProvider'])
    scale = float(np.abs(cpu).max())
    rel = float(np.abs(gpu - cpu).max()) / scale
    rel_exact = float(np.abs(exact.run(None, {'x': x})[0] - cpu).max()) / scale
    ok = rel < 1e-2 and rel_exact < 1e-4
    print(f'conv + matmul on the GPU vs CPU, max difference relative to the largest value: '
          f'{rel:.1e} (TF32, as the app runs), {rel_exact:.1e} (TF32 off)')
    print('RESULT: ' + ('OK, ONNX Runtime runs on the GPU' if ok else 'FAILED, the GPU gives wrong numbers'))
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
