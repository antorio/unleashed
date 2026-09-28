import numpy as np
import cv2
import onnxruntime
import unleashed.globals

from unleashed.typing import Frame
from unleashed.utilities import resolve_relative_path


class FFXSegBase:
    """Shared logic for FaceFusion's DeepFaceLab-XSeg occluder models.

    These are the same XSeg architecture as the stock xseg.onnx (256x256 input,
    /255 normalisation, NHWC), just different trained weights. Subclasses set
    `model_files` (a list of .onnx filenames under ../models/) and `processorname`.

    With more than one model file the raw face masks are intersected
    (numpy.minimum) -- a pixel counts as 'face' only if every model agrees,
    so any occluder flagged by any model is masked out. This is the broadest
    occlusion coverage ('combined' mode), matching FaceFusion's `many` mode.

    NOTE: matches the stock Mask_XSeg preprocessing exactly (BGR, no channel
    swap). If a model's mask ever looks wrong, the usual culprit is a BGR/RGB
    mismatch -- add a cv2.cvtColor(...BGR2RGB) before normalisation.
    """
    plugin_options: dict = None
    type = 'mask'
    model_files: list = []      # overridden by subclass
    processorname = None        # overridden by subclass
    sessions = None

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.sessions is None:
            onnxruntime.set_default_logger_severity(3)
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            self.sessions = []
            for fn in self.model_files:
                model_path = resolve_relative_path('../models/' + fn)
                sess = onnxruntime.InferenceSession(
                    model_path, None, providers=__import__('unleashed.utilities',fromlist=['tuned_execution_providers']).tuned_execution_providers())
                self.sessions.append((sess, sess.get_inputs(), sess.get_outputs()))

    def Run(self, img1, keywords: str) -> Frame:
        temp_frame = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_CUBIC)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = temp_frame[None, ...]

        face = None
        for (sess, ins, outs) in self.sessions:
            io_binding = sess.io_binding()              # per-call -> thread-safe
            io_binding.bind_cpu_input(ins[0].name, temp_frame)
            io_binding.bind_output(outs[0].name, "cpu")      # host output: see FaceSwapInsightFace.Run
            sess.run_with_iobinding(io_binding)
            r = io_binding.copy_outputs_to_cpu()[0][0]
            r = np.clip(r, 0, 1.0)
            # intersect the face masks -> broadest occlusion across the variants
            face = r if face is None else np.minimum(face, r)

        face[face < 0.1] = 0
        # invert: high = area to keep from the original (occluder), as Mask_XSeg
        return 1.0 - face

    def Release(self):
        self.sessions = None
