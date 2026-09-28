import unleashed.globals
import numpy as np
import onnx
import onnxruntime

from unleashed.typing import Face, Frame
from unleashed.utilities import resolve_relative_path



class FaceSwapInsightFace():
    plugin_options:dict = None
    model_swap_insightface = None

    processorname = 'faceswap'
    type = 'swap'


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"] or self.plugin_options["modelname"] != plugin_options["modelname"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_swap_insightface is None:
            model_path = resolve_relative_path('../models/' + self.plugin_options["modelname"])
            graph = onnx.load(model_path).graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            self.input_mean = 0.0
            self.input_std = 255.0
            #cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}            
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False
            from unleashed.utilities import tuned_execution_providers
            provs = tuned_execution_providers()
            print(f"[load] Creating face-swap (inswapper) session: {self.plugin_options['modelname']}  providers={provs}")
            self.model_swap_insightface = onnxruntime.InferenceSession(model_path, sess_options, providers=provs)



    def identity_latent(self, source_face: Face, target_face: Face):
        """inswapper's source input: the source identity projected through emap.

        Identity strength (unleashed.globals.identity_strength, 0 = off) pushes the
        source identity away from the target's own identity before projecting:
            e = normalize(src + w * (src - tgt))
        inswapper keeps part of the target's identity in its output; steering
        the conditioning away from it strengthens the source (inner face). Both
        vectors are unit ArcFace embeddings, i.e. the same space. tgt is the
        target's embedding smoothed over the video (identity_ref, set by the
        landmark stabilizer) when available, so the push does not flicker."""
        e = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape(1, -1)
        w = float(getattr(unleashed.globals, 'identity_strength', 0.0) or 0.0)
        if w > 0.0 and target_face is not None:
            t = target_face.get('identity_ref')
            if t is None:
                t = target_face.get('embedding')
            if t is not None:
                t = np.asarray(t, dtype=np.float32).reshape(1, -1)
                t = t / (np.linalg.norm(t) + 1e-9)
                e = e + w * (e - t)
                e = e / (np.linalg.norm(e) + 1e-9)
        latent = np.dot(e, self.emap)
        latent /= np.linalg.norm(latent)
        return latent.astype(np.float32)


    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        latent = self.identity_latent(source_face, target_face)
        io_binding = self.model_swap_insightface.io_binding()           
        io_binding.bind_cpu_input("target", temp_frame)
        io_binding.bind_cpu_input("source", latent)
        # the output straight to host memory: left on the GPU and copied
        # after the run, it came back wrong now and then when threads share
        # the session (L4: 1 of 1600 calls, off by up to 0.97 -- a ruined
        # slice of the face; 0 of 1600 this way, same speed; tools/ort_race.py)
        io_binding.bind_output("output", "cpu")
        self.model_swap_insightface.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()[0]
        return ort_outs[0]


    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None


                



