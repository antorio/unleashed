import unleashed.globals
import numpy as np
import onnx
import onnxruntime

from unleashed.typing import Face, Frame
from unleashed.utilities import resolve_relative_path

# Identity boost: how far the source vector is pushed past what inswapper
# made (0.5 measured best; 1.0 and a second step overshoot, see boosted_latent)
IDENTITY_BOOST_WEIGHT = 0.5


def _recogniser():
    """buffalo_l's recognition model (w600k), the one the source embeddings
    come from; None if the analyser was built without it."""
    try:
        from unleashed.face_util import get_face_analyser
        return get_face_analyser().models.get('recognition')
    except Exception:
        return None


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



    def source_latent(self, source_face: Face):
        """inswapper's source input: the source identity projected through emap
        (exactly as before the identity boost existed)."""
        e = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape(1, -1)
        latent = np.dot(e, self.emap)
        latent /= np.linalg.norm(latent)
        return latent.astype(np.float32)


    def boosted_latent(self, source_face: Face, crop: Frame):
        """Identity boost (unleashed.globals.identity_boost): inswapper does not
        reproduce the source fully, so swap once, read the identity of what it
        actually made (o, buffalo_l w600k on the result) and push the source
        vector away from that miss:  e = normalize(s + w * (s - o)), w = 0.5.

        Measured (tools/identity_check.py / identity_variants.py, 30 pairs from
        t1.jpg, judged by antelopev2 glintr100): likeness to the source 0.790 ->
        0.809, no pair worse. The old formula (push away from the TARGET's
        identity) lowered it at every strength (0.738 at 1.0) and was removed.

        crop: one prepared 128 crop (with pixel boost the first slice): the
        correction is computed ONCE per face and reused for every slice and
        pass, so the extra cost is one 128 swap + one recognition per face.
        Returns None (= plain source latent) if no recogniser is available."""
        rec = _recogniser()
        if rec is None:
            return None
        s = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape(1, -1)
        s = s / (np.linalg.norm(s) + 1e-9)
        out = self._run(self.source_latent(source_face), crop)
        # inswapper's 128 crop is the arcface 112 template shifted 8 px right
        img = np.clip(out.transpose(1, 2, 0)[:, :, ::-1] * 255.0, 0, 255).round().astype(np.uint8)
        o = np.asarray(rec.get_feat(np.ascontiguousarray(img[0:112, 8:120])), dtype=np.float32).reshape(1, -1)
        o = o / (np.linalg.norm(o) + 1e-9)
        e = s + IDENTITY_BOOST_WEIGHT * (s - o)
        latent = np.dot(e / (np.linalg.norm(e) + 1e-9), self.emap)
        latent /= np.linalg.norm(latent)
        return latent.astype(np.float32)


    def _run(self, latent, temp_frame):
        io_binding = self.model_swap_insightface.io_binding()
        io_binding.bind_cpu_input("target", temp_frame)
        io_binding.bind_cpu_input("source", latent)
        # the output straight to host memory: left on the GPU and copied
        # after the run, it came back wrong now and then when threads share
        # the session (L4: 1 of 1600 calls, off by up to 0.97 -- a ruined
        # slice of the face; 0 of 1600 this way, same speed; tools/ort_race.py)
        io_binding.bind_output("output", "cpu")
        self.model_swap_insightface.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()[0][0]


    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame, latent=None) -> Frame:
        """latent: a precomputed source input (boosted_latent), else the plain one."""
        if latent is None:
            latent = self.source_latent(source_face)
        return self._run(latent, temp_frame)


    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None


                



