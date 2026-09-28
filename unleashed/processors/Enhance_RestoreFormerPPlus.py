from typing import Any, List, Callable
import cv2 
import numpy as np
import onnxruntime
import unleashed.globals

from unleashed.typing import Face, Frame, FaceSet
from unleashed.utilities import resolve_relative_path

class Enhance_RestoreFormerPPlus():
    plugin_options:dict = None
    model_restoreformerpplus = None
    devicename = None
    name = None

    processorname = 'restoreformer++'
    type = 'enhance'
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_restoreformerpplus is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            model_path = resolve_relative_path('../models/restoreformer_plus_plus.onnx')
            self.model_restoreformerpplus = onnxruntime.InferenceSession(model_path, None, providers=__import__('unleashed.utilities',fromlist=['tuned_execution_providers']).tuned_execution_providers())
            self.model_inputs = self.model_restoreformerpplus.get_inputs()
            self.model_outputs = self.model_restoreformerpplus.get_outputs()

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]
        temp_frame = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)
        
        # Fresh io_binding per call (thread-safe): a shared binding races across
        # the worker threads and produces stale/jittered frames every so often.
        io_binding = self.model_restoreformerpplus.io_binding()
        io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        io_binding.bind_output(self.model_outputs[0].name, "cpu")   # host output: see FaceSwapInsightFace.Run
        self.model_restoreformerpplus.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs 
        
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        # float, not int: the enhancer always outputs 512, so with a swap
        # crop larger than that (subsample 768/1024) int() truncated the
        # ratio to 0. paste_upscale then does M_scale = M * scale_factor,
        # zeroing the whole affine matrix -> degenerate inverse -> the
        # swap smeared over the frame / rendered black.
        scale_factor = float(result.shape[1]) / float(input_size)
        return result.astype(np.uint8), scale_factor


    def Release(self):
        del self.model_restoreformerpplus
        self.model_restoreformerpplus = None

