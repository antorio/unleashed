import os
import cv2 
import numpy as np
import psutil

from unleashed.ProcessOptions import ProcessOptions

from unleashed.face_util import get_first_face, get_first_face_multi, get_all_faces_multi, rotate_anticlockwise, rotate_clockwise, clamp_cut_values
from unleashed.landmark68 import refine_faces_landmark68
from unleashed.utilities import compute_cosine_distance, get_device, str_to_class, shuffle_array
from unleashed.face_stabilizer import LandmarkStabilizer, TargetTracks

from typing import Any, List, Callable
from unleashed.typing import Frame, Face
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Thread, Lock, Condition
from contextlib import contextmanager
import time
from queue import Queue, Full
from tqdm import tqdm
from unleashed.ffmpeg_writer import FFMPEG_VideoWriter
import unleashed.globals



# Poor man's enum to be able to compare to int
class eNoFaceAction():
    USE_ORIGINAL_FRAME = 0
    RETRY_ROTATED = 1
    SKIP_FRAME = 2
    # NOTE: this used to be `3,` (a stray comma made it the TUPLE (3,)), so the
    # int comparisons `no_face_action == SKIP_FRAME_IF_DISSIMILAR` were always
    # False and the mode silently behaved like Retry-rotated. Fixed to int.
    SKIP_FRAME_IF_DISSIMILAR = 3
    USE_LAST_SWAPPED = 4



def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues


# swap_faces() result when the detector found no face at all (at any of the
# tried angles), as opposed to 0 = faces found but none of them swapped.
# Every caller tests "> 0", so it reads as "nothing swapped" everywhere.
NO_FACE_DETECTED = -1


# Idle processors (with their loaded ONNX sessions), kept between runs.
# Preview -> render -> preview used to release every session and build it
# again (plus cuDNN's EXHAUSTIVE algorithm search on its first inference) each
# time. ProcessMgr itself stays per-run, so its per-run state (stabilizer,
# sequencer, counters) is still fresh; only the model wrappers are shared, and
# they hold no per-frame state.
_idle_processors = {}
_idle_lock = Lock()


def _plugin_key(processor):
    # the key from ProcessMgr.plugins; processorname does not always match it
    # (Clip2Seg: 'clip2seg' vs 'mask_clip2seg'), which made that model reload
    # on every initialize
    return getattr(processor, 'plugin_key', processor.processorname)


def _take_idle_processor(key):
    with _idle_lock:
        return _idle_processors.pop(key, None)


def _release_idle_processors(keep=()):
    with _idle_lock:
        drop = [key for key in _idle_processors if key not in keep]
        processors = [_idle_processors.pop(key) for key in drop]
    for p in processors:
        p.Release()



class FrameSequencer():
    """Lets frames enter a critical section in FRAME ORDER.

    Frames are handed to workers round-robin (frame N -> queue[N % threads]) and
    all workers run at once, so the shared LandmarkStabilizer was receiving
    frames in an arbitrary, interleaved order -- its "previous frame" could be
    any of the 8 in flight. Temporal smoothing on a shuffled sequence is not
    smoothing at all: it blended each frame against the wrong neighbour and its
    motion-adaptive alpha measured motion across ~8-frame gaps, so it kept
    concluding "big movement" and disengaging.

    Only landmark refinement + smoothing run inside the section; detection and
    the expensive swap/paste stay fully parallel. While frame N is inside the
    section, frame N-1 is already swapping, so throughput is barely affected.

    Every video frame must take its turn exactly once, including frames in
    which no face was found (pass_turn). A frame that never checks in makes the
    next frame wait for the full timeout -- a 10 s freeze per face dropout.
    """
    def __init__(self):
        self._cond = Condition()
        self._expected = 0

    def reset(self):
        with self._cond:
            self._expected = 0
            self._cond.notify_all()

    def pass_turn(self, index):
        """Take this frame's turn without doing any ordered work."""
        with self.in_order(index):
            pass

    @contextmanager
    def in_order(self, index, timeout=10.0):
        if index is None:
            yield          # no index (preview / image batch) -> no ordering
            return
        with self._cond:
            deadline = time.time() + timeout
            # Wait only while an EARLIER frame still has to pass. A frame that is
            # already late (a later frame timed out past it) goes straight in:
            # its turn can never come round again, so waiting would only stall.
            while self._expected < index and unleashed.globals.processing:
                remaining = deadline - time.time()
                if remaining <= 0:
                    # Never hang the render: if the frame we are waiting for was
                    # dropped, carry on unordered rather than deadlock.
                    break
                # wait in short slices: an abort flips unleashed.globals.processing
                # without touching this condition, so a waiter that slept for the
                # whole timeout would keep the Stop button hanging for seconds.
                self._cond.wait(min(remaining, 0.1))
        try:
            yield
        finally:
            with self._cond:
                # always advance, even on exception, or every later frame stalls
                if index >= self._expected:
                    self._expected = index + 1
                self._cond.notify_all()


class FrameOrderGate():
    """wait(N) returns once frames 0..N-1 have called done(); frames may call
    done() in any order. Used by the Expression Restorer's temporal smoother,
    which runs late in a frame (after the swap), outside FrameSequencer's
    section: with 8 workers it was fed frames in whatever order they
    finished. Separate from FrameSequencer, which it does not change.

    Frame N waits only for EARLIER frames, and a worker takes its frames in
    increasing order, so a wait can never depend on a frame queued behind
    it. Stop (processing False) and a long timeout end any wait.

    A frame signals done() right after the smoothing of its last face, so
    the next frames wait for that step only, not for the generator, mask,
    enhancer and paste that follow it; the end of the frame signals again
    (faces without a smoothing step, errors)."""
    def __init__(self):
        self._cond = Condition()
        self._next = 0
        self._done = set()

    def reset(self):
        with self._cond:
            self._next = 0
            self._done.clear()
            self._cond.notify_all()

    def wait(self, index, timeout=300.0):
        if index is None:
            return
        with self._cond:
            deadline = time.time() + timeout
            while self._next < index and unleashed.globals.processing:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._cond.wait(min(remaining, 0.1))

    def done(self, index):
        """Idempotent: a frame may signal early (right after its last
        smoothing) and again when it ends."""
        if index is None:
            return
        with self._cond:
            if index < self._next:
                return
            self._done.add(index)
            while self._next in self._done:
                self._done.discard(self._next)
                self._next += 1
            self._cond.notify_all()


# Two picks this close are the same person (picked from two frames). It is the
# old default tolerance: what the app has always counted as "that person". In
# the test scene one person's picks were <= 0.44 apart, different people >= 0.78.
SAME_PICK_DISTANCE = 0.65


def match_selected_faces(target_faces, faces, threshold):
    """Specific people: which detected faces to swap, as indices into `faces`.

    Each pick (a picked person) takes the detected face closest to it within
    the tolerance (cosine distance <= threshold); picks with the closest match
    go first. A pick whose closest face is already taken
      - by a pick of the same person (the same person picked twice, from
        two frames: the picks are within SAME_PICK_DISTANCE, or within a
        tighter tolerance) takes nothing, so it does not grab somebody else's
        face;
      - by a clearly different pick, takes its next closest free face within
        the tolerance.
    The result is in the order of the picks, the order the faces were
    processed in before.

    It used to take the LEFT-MOST free face within the tolerance: with a
    loose tolerance a different person further left was swapped (6-person
    clip at 0.8: the wrong person in 29 of 29 frames; 0 at 0.65), and a second
    pick of the same person took somebody else's face. When each pick has at
    most one face within the tolerance and no face is within it for two picks,
    the choice is the same as before."""
    dist = [[float(compute_cosine_distance(tf.embedding, face.embedding)) for face in faces]
            for tf in target_faces]
    order = sorted((i for i in range(len(target_faces)) if dist[i]),
                   key=lambda i: (min(dist[i]), i))
    owner = {}                              # face index -> pick index
    for ti in order:
        for fi in sorted(range(len(faces)), key=lambda f: (dist[ti][f], f)):
            if dist[ti][fi] > threshold:
                break
            other = owner.get(fi)
            if other is None:
                owner[fi] = ti
                break
            same_person = float(compute_cosine_distance(target_faces[ti].embedding, target_faces[other].embedding)) \
                <= min(threshold, SAME_PICK_DISTANCE)
            if same_person:
                break
    claimed = {ti: fi for fi, ti in owner.items()}
    return [claimed[ti] for ti in sorted(claimed)]


class ProcessMgr():
    input_face_datas = []
    target_face_datas = []

    imagemask = None

    processors = []
    options : ProcessOptions = None
    
    num_threads = 1
    current_index = 0
    processing_threads = 1
    buffer_wait_time = 0.1

    lock = Lock()

    frames_queue = None
    processed_queue = None

    videowriter= None
    streamwriter = None

    progress_gradio = None
    _progress_gradio_lock = Lock()
    total_frames = 0

    num_frames_no_face = 0
    last_swapped_frame = None

    output_to_file = None
    output_to_cam = None

    # video-only landmark smoothing
    stabilizer = None
    video_mode = False


    plugins =  { 
    'faceswap'          : 'FaceSwapInsightFace',
    'mask_clip2seg'     : 'Mask_Clip2Seg',
    'mask_xseg'         : 'Mask_XSeg',
    'mask_ffxseg1'      : 'Mask_FFXSeg1',
    'mask_ffxseg2'      : 'Mask_FFXSeg2',
    'mask_ffxseg3'      : 'Mask_FFXSeg3',
    'mask_ffxseg_many'  : 'Mask_FFXSegMany',
    'mask_faceparser'   : 'Mask_FaceParser',
    'expression_restorer' : 'Expression_LivePortrait',
    'codeformer'        : 'Enhance_CodeFormer',
    'gfpgan'            : 'Enhance_GFPGAN',
    'restoreformer++'   : 'Enhance_RestoreFormerPPlus',
    'filter_generic'    : 'Frame_Filter',
    'removebg'          : 'Frame_Masking',
    }

    def __init__(self, progress):
        if progress is not None:
            self.progress_gradio = progress
            # gr.Progress finds its event through context variables, which the
            # frame worker threads do not inherit: called from a worker it did
            # nothing, so the UI never showed a render's progress. Its updates
            # run in a copy of this (the event's) context instead.
            import contextvars
            self._progress_context = contextvars.copy_context()

    def reuseOldProcessor(self, name:str):
        for p in self.processors:
            if _plugin_key(p) == name:
                return p

        return None


    def initialize(self, input_faces, target_faces, options, release_idle=True):
        # release_idle=False (the Face Swap tab's Mask view): leave the idle
        # models of a finished render alone, the next preview takes them.
        # Own copies of the face lists: the UI can add/remove faces while a
        # render runs, and the render must keep using the faces it started with.
        self.input_face_datas = list(input_faces)
        self.target_face_datas = list(target_faces)
        self.num_frames_no_face = 0
        self.last_swapped_frame = None
        self.options = options
        self._pipeline_error = None
        # video_mode is set by run_batch_inmem for its video only: it used to
        # stay on for good, so a later Start's image batch got the landmark
        # smoothing across unrelated images
        self.video_mode = False
        self.er_order = FrameOrderGate()
        self._tls = threading.local()           # the frame index a worker is on
        devicename = get_device()

        # (re)create the temporal landmark smoother for this run
        self.sequencer = FrameSequencer()
        self.stabilizer = LandmarkStabilizer(
            strength=unleashed.globals.landmark_smoothing_strength,
            deadzone_frac=getattr(unleashed.globals, 'landmark_smoothing_deadzone', 0.0)
        )
        # running identity / face-shape averages for the likeness options
        self.target_tracks = TargetTracks()
        # NOTE: the One-Euro smoothing of the alignment matrix M (batch 2/3) was
        # REMOVED entirely: even with a pixel deadband it produced visible
        # micro-jitter on footage whose alignment was already steady (user
        # confirmed A/B). Landmark smoothing above is the only temporal filter.

        # Face-analysis module set. Kept CONSTANT across every swap mode so the
        # analyser is never rebuilt mid-session (a different module set yields
        # subtly different landmark_2d_106, which drives the Expression Restorer
        # and made the same ER strength look different between modes).
        #
        # buffalo_l's 68-point model (1k3d68.onnx) is requested ONLY when its
        # output is actually consumed:
        #   - alignment off            -> nobody reads the 68 points at all
        #   - alignment on + 2dfan4 on -> 2dfan4 OVERWRITES them; the only thing
        #     kept from buffalo was the z column, and NOTHING reads z
        #     (landmark_68_to_5 uses [:, :2], the stabilizer leaves z untouched).
        # In both cases running 1k3d68 on every face of every frame is inference
        # that is paid for and thrown away.
        analysis_modules = ["landmark_2d_106", "detection", "recognition", "genderage"]
        align_68 = getattr(unleashed.globals, 'use_landmark_alignment', True)
        hi_68 = getattr(unleashed.globals, 'use_hi_landmarker', False)
        # face shape from the source compares shapes in 3D: it always needs
        # buffalo's 68 points (with depth), 2dfan4 or not
        shape_68 = float(getattr(unleashed.globals, 'face_shape_strength', 0.0) or 0.0) > 0.0
        if (align_68 and not hi_68) or shape_68:
            analysis_modules.insert(0, "landmark_3d_68")
        unleashed.globals.g_desired_face_analysis = analysis_modules
        if options.swap_mode == "all_random":
            # don't modify original list
            self.input_face_datas = input_faces.copy()
            shuffle_array(self.input_face_datas)


        # Release processors that are no longer requested -- ours and idle ones
        # left by a previous run -- before creating any new one, so peak GPU
        # memory stays what it was.
        for p in self.processors:
            if _plugin_key(p) not in options.processors:
                p.Release()
        if release_idle:
            _release_idle_processors(keep=options.processors.keys())

        newprocessors = []
        for key, extoption in options.processors.items():
            p = self.reuseOldProcessor(key)
            if p is None:
                p = _take_idle_processor(key)
            if p is None:
                classname = self.plugins[key]
                module = 'unleashed.processors.' + classname
                p = str_to_class(module, classname)
            if p is not None:
                p.plugin_key = key
                extoption.update({"devicename": devicename})
                if p.type == "swap":
                    # ReSwapper removed -- InSwapper 128 is the only swap model.
                    extoption.update({"modelname": "inswapper_128.onnx"})

                p.Initialize(extoption)
                newprocessors.append(p)
            else:
                print(f"Not using {module}")
        self.processors = newprocessors

        # Temporal smoother for the LivePortrait expression vector (reset per run).
        # Created AFTER the processors so a smoother problem can never prevent the
        # mask/other processors from being built. Non-fatal.
        self.exp_smoother = None
        try:
            from unleashed.expression_smoother import ExpressionSmoother
            self.exp_smoother = ExpressionSmoother(
                strength=getattr(unleashed.globals, 'expression_smoothing_strength', 0.0)
            )
        except Exception as _e:
            print(f"[expr-smoother] init skipped ({_e})")




        self.options.imagemask = self.manual_mask(self.options.imagemask)
        self._imagemask_fitted = None

        self.options.frame_processing = False
        for p in self.processors:
            if p.type.startswith("frame_"):
                self.options.frame_processing = True

            
 



    def run_batch(self, source_files, target_files, threads:int = 1):
        self.video_mode = False                 # images / extracted frames: no temporal state
        progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        self.total_frames = len(source_files)
        self.num_threads = threads
        with tqdm(total=self.total_frames, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = []
                queue = create_queue(source_files)
                queue_per_future = max(len(source_files) // threads, 1)
                while not queue.empty():
                    future = executor.submit(self.process_frames, source_files, target_files, pick_queue(queue, queue_per_future), lambda: self.update_progress(progress))
                    futures.append(future)
                for future in as_completed(futures):
                    future.result()


    def process_frames(self, source_files: List[str], target_files: List[str], current_files, update: Callable[[], None]) -> None:
        # source->target lookup built once: the old per-frame source_files.index(f)
        # was O(n) per frame (O(n^2) for long extract-frames batches).
        target_map = dict(zip(source_files, target_files))
        for f in current_files:
            if not unleashed.globals.processing:
                return
            
            # Decode the byte array into an OpenCV image
            temp_frame = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
            if temp_frame is not None:
                if self.options.frame_processing:
                    # chain the processors (frame = p.Run(frame)); the old code
                    # fed temp_frame to every processor so only the LAST one's
                    # effect survived (process_videoframes already chained).
                    frame = temp_frame
                    for p in self.processors:
                        frame = p.Run(frame)
                    resimg = frame
                else:
                    resimg = self.process_frame(temp_frame)
                if resimg is not None:
                    # Also let numpy write the file to support utf-8/16 filenames
                    cv2.imencode(f'.{unleashed.globals.CFG.output_image_format}',resimg)[1].tofile(target_map[f])
            if update:
                update()



    def read_frames_thread(self, cap, frame_start, frame_end, num_threads):
        num_frame = 0
        total_num = frame_end - frame_start
        if frame_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_start)

        while unleashed.globals.processing:
            ret, frame = cap.read()
            if not ret:
                break

            # put with a short timeout and re-check processing: if a worker died
            # (its queue is never drained) this loop can still notice the run was
            # aborted and exit, instead of blocking forever on a full queue and
            # deadlocking the whole teardown.
            while unleashed.globals.processing:
                try:
                    self.frames_queue[num_frame % num_threads].put((num_frame, frame), block=True, timeout=0.2)
                    break
                except Full:
                    continue
            num_frame += 1
            if num_frame == total_num:
                break

        # Sentinels MUST always be delivered (blocking, no timeout): if one is
        # dropped, its worker never receives None, never emits its (False, None),
        # and the write thread waits on that producer forever -> join() hangs and
        # the finished video is never muxed/saved. A brief block here is fine --
        # the workers keep draining, so the queue clears within a frame or two.
        for i in range(num_threads):
            self.frames_queue[i].put(None, block=True)



    def process_videoframes(self, threadindex, progress) -> None:
        while True:
            item = self.frames_queue[threadindex].get()
            if item is None:
                self.processing_threads -= 1
                self.processed_queue[threadindex].put((False, None))
                return
            else:
                # the read thread tags every frame with its index so the
                # temporal section can run in true frame order (FrameSequencer)
                frame_index, frame = item
                if not unleashed.globals.processing:
                    # stopped: drop the frames still queued instead of swapping
                    # them first. The placeholder keeps the writer's round robin
                    # over the per-thread queues aligned (it skips (True, None)).
                    self.processed_queue[threadindex].put((True, None))
                    del frame
                    continue
                failed = False
                try:
                    if self.options.frame_processing:
                        for p in self.processors:
                            frame = p.Run(frame)
                        resimg = frame
                    else:
                        resimg = self.process_frame(frame, frame_index)
                except Exception as e:
                    failed = True
                    # One frame failed (e.g. out of GPU memory). Stop the run,
                    # but keep this worker draining its queue: a worker that
                    # dies here leaves the others blocked on their queues and
                    # the render hangs for good (Stop cannot free it). The
                    # error is raised once the pipeline has wound down.
                    import traceback as _tb
                    _tb.print_exc()
                    if self._pipeline_error is None:
                        self._pipeline_error = e
                    unleashed.globals.processing = False
                    resimg = None
                self.processed_queue[threadindex].put((True, resimg))
                del frame
                if not failed:
                    progress()          # frames dropped by "Drop the frame" count too


    def write_frames_thread(self):
        nextindex = 0
        num_producers = self.num_threads
        write_failed = False

        while True:
            process, frame = self.processed_queue[nextindex % self.num_threads].get()
            nextindex += 1
            if frame is not None:
                if not write_failed:
                    try:
                        if self.output_to_file:
                            self.videowriter.write_frame(frame)
                        if self.output_to_cam:
                            self.streamwriter.WriteToStream(frame)
                    except Exception as e:
                        # ffmpeg gone (codec that does not fit the format, disk
                        # full ...): stop the run but keep taking frames until
                        # every worker has finished, or they block on their
                        # queues forever
                        import traceback as _tb
                        _tb.print_exc()
                        write_failed = True
                        if self._pipeline_error is None:
                            self._pipeline_error = e
                        unleashed.globals.processing = False
                del frame
            elif process == False:
                num_producers -= 1
                if num_producers < 1:
                    return
            


    def run_batch_inmem(self, output_method, source_video, target_video, frame_start, frame_end, fps, threads:int = 1):
        if len(self.processors) < 1:
            print("No processor defined!")
            return

        # streaming video path -> enable temporal landmark smoothing
        self.video_mode = True
        if self.stabilizer is not None:
            self.stabilizer.reset()
        if getattr(self, 'target_tracks', None) is not None:
            self.target_tracks.reset()
        if getattr(self, 'sequencer', None) is not None:
            self.sequencer.reset()
        # the Expression Restorer's smoother: this video only (it carried the
        # expressions of the previous file of the run)
        if getattr(self, 'exp_smoother', None) is not None:
            self.exp_smoother.tracks = []
        self.er_order.reset()

        cap = cv2.VideoCapture(source_video)
        # endframe is a COUNT (get_video_frame_total), so the range processed is
        # [frame_start, frame_end) == frame_end - frame_start frames -- exactly
        # what read_frames_thread reads. The old "+1" made the progress target
        # one larger than the frames ever produced, so the bar stuck at N/(N+1),
        # the Gradio generator never reached 100%, the UI stayed "processing"
        # after the video finished, and Stop then returned a non-JSON response
        # (the "JSON.parse ... unexpected character" error). Matching the counts
        # lets the run reach 100% and the button flip back to idle on its own.
        frame_count = (frame_end - frame_start)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        processed_resolution = None
        for p in self.processors:
            if hasattr(p, 'getProcessedResolution'):
                processed_resolution = p.getProcessedResolution(width, height)
                print(f"Processed resolution: {processed_resolution}")
        if processed_resolution is not None:
            width = processed_resolution[0]
            height = processed_resolution[1]


        self.total_frames = frame_count
        self.num_threads = threads
        self._pipeline_error = None             # first error of a frame or of the writer

        self.processing_threads = self.num_threads
        self.frames_queue = []
        self.processed_queue = []
        for _ in range(threads):
            self.frames_queue.append(Queue(1))
            self.processed_queue.append(Queue(1))

        self.output_to_file = output_method != "Virtual Camera"
        self.output_to_cam = output_method == "Virtual Camera" or output_method == "Both"

        if self.output_to_file:
            # frames go in at the source's rate; a different output rate is
            # resampled by ffmpeg (frames dropped / repeated, same duration, so
            # the audio still fits) instead of re-timing every frame
            from unleashed.utilities import detect_fps
            src_fps = float(detect_fps(source_video) or fps)
            out_fps = fps if abs(float(fps) - src_fps) > 0.01 else None
            self.videowriter = FFMPEG_VideoWriter(target_video, (width, height), src_fps if out_fps else fps, codec=unleashed.globals.video_encoder,
                                                  crf=unleashed.globals.video_quality, audiofile=None, out_fps=out_fps)
        if self.output_to_cam:
            from unleashed.StreamWriter import StreamWriter
            self.streamwriter = StreamWriter((width, height), int(fps))

        # daemon=True: even in a worst-case teardown failure these can never
        # keep the process (or a future run) hostage.
        readthread = Thread(target=self.read_frames_thread, args=(cap, frame_start, frame_end, threads), daemon=True)
        readthread.start()

        writethread = Thread(target=self.write_frames_thread, daemon=True)
        writethread.start()

        worker_error = None
        progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        with tqdm(total=self.total_frames, desc='Processing', unit='frames', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
            with ThreadPoolExecutor(thread_name_prefix='swap_proc', max_workers=self.num_threads) as executor:
                futures = []
                
                for threadindex in range(threads):
                    future = executor.submit(self.process_videoframes, threadindex, lambda: self.update_progress(progress))
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        # A dead worker stops consuming its frames_queue -> the
                        # read thread blocks forever on put() -> sentinels never
                        # go out -> every other thread deadlocks -> the UI stays
                        # on "processing" forever. Record the error, stop the
                        # pipeline, and fall through to the guarded teardown.
                        if worker_error is None:
                            worker_error = e
                        unleashed.globals.processing = False
                        import traceback as _tb
                        _tb.print_exc()
        # ---- teardown ----
        # CRITICAL: on the normal (no-error) path we must wait for the threads to
        # finish for real -- the write thread may still be muxing the last frames.
        # Joining with a timeout here and moving on would close the videowriter
        # mid-write and the finished video would never reach the output folder.
        # Timeouts are used ONLY when a worker actually crashed (then blocking
        # forever is the failure we're avoiding). Success path == original
        # behaviour (unbounded joins), so a completed render is always saved.
        if worker_error is None:
            readthread.join()
            writethread.join()
        else:
            # A crashed worker stopped draining its queue; unblock the read
            # thread (which may be stuck on a full queue) then feed the write
            # thread enough sentinels to let it finish what it can and exit.
            unleashed.globals.processing = False
            for _ in range(600):                      # <= ~60s
                if not readthread.is_alive():
                    break
                for q in self.frames_queue:
                    try:
                        q.get_nowait()
                    except Exception:
                        pass
                readthread.join(timeout=0.1)
            for q in self.processed_queue:
                try:
                    q.put((False, None), block=True, timeout=1)
                except Exception:
                    pass
            writethread.join(timeout=60)
            if readthread.is_alive() or writethread.is_alive():
                print("[finish] WARNING: a worker crashed and a thread did not exit cleanly (leaked as daemon)")

        cap.release()
        if self.output_to_file:
            try:
                self.videowriter.close()
            except Exception as e:
                # a writer whose ffmpeg already died cannot be closed cleanly
                if self._pipeline_error is None:
                    self._pipeline_error = e
        if self.output_to_cam:
            self.streamwriter.Close()

        self.frames_queue.clear()
        self.processed_queue.clear()
        # Surface a worker error only AFTER the writer is closed and whatever
        # frames did get processed are flushed to the file, so a partial video
        # still lands in the output folder (user preference: a result WITH an
        # error beats a clean failure with no result).
        pipeline_error, self._pipeline_error = self._pipeline_error, None
        if worker_error is not None:
            raise worker_error
        if pipeline_error is not None:
            raise pipeline_error




    def update_progress(self, progress: Any = None) -> None:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        progress.set_postfix({
            'memory_usage': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',
            'execution_threads': self.num_threads
        })
        progress.update(1)
        if self.progress_gradio is not None:
            # The Gradio tracker talks to the browser over the event stream; on
            # Colab that stream goes through the gradio.live tunnel and can die
            # mid-render. A raising tracker must NEVER kill a worker thread (a
            # dead worker deadlocks the read/write queue pipeline and leaves the
            # UI stuck on "processing" forever). Also serialize the calls: 8
            # workers pushing progress concurrently is not guaranteed safe.
            try:
                with self._progress_gradio_lock:          # one thread at a time in the copied context
                    context = getattr(self, '_progress_context', None)
                    update = (progress.n, self.total_frames)
                    if context is not None:
                        context.run(self.progress_gradio, update, desc='Processing', total=self.total_frames, unit='frames')
                    else:
                        self.progress_gradio(update, desc='Processing', total=self.total_frames, unit='frames')
            except Exception as e:
                print(f"[finish] gradio progress tracker failed ({e}); continuing render without UI progress")
                self.progress_gradio = None



    def process_frame(self, frame:Frame, frame_index=None):
        self._tls.frame_index = frame_index
        self._tls.er_faces_left = None          # set by swap_faces
        try:
            result = self._process_frame(frame, frame_index)
            # the painted keep-original mask: on the final upright frame (after a
            # rotated retry), against the original frame
            if result is not None and getattr(self.options, 'imagemask', None) is not None:
                result = self.simple_blend_with_mask(result, frame, self.fitted_imagemask(frame))
            return result
        finally:
            # every video frame passes the Expression Restorer's order gate
            # exactly once, faces or not, error or not
            self._tls.frame_index = None
            self._tls.er_faces_left = None
            self.er_order.done(frame_index)


    def _process_frame(self, frame:Frame, frame_index=None):
        if len(self.input_face_datas) < 1 and not self.options.show_face_masking:
            return frame
        temp_frame = frame.copy()
        num_swapped, temp_frame = self.swap_faces(frame, temp_frame, frame_index)
        if num_swapped > 0:
            if unleashed.globals.no_face_action == eNoFaceAction.SKIP_FRAME_IF_DISSIMILAR:
                if len(self.input_face_datas) > num_swapped:
                    return None
            self.num_frames_no_face = 0
            # only "Use last swapped" ever reads this copy of the whole frame
            if unleashed.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
                self.last_swapped_frame = temp_frame.copy()
            return temp_frame
        if unleashed.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
            if self.last_swapped_frame is not None and self.num_frames_no_face < self.options.max_num_reuse_frame:
                self.num_frames_no_face += 1
                return self.last_swapped_frame.copy()
            return frame

        elif unleashed.globals.no_face_action == eNoFaceAction.USE_ORIGINAL_FRAME:
            return frame
        if unleashed.globals.no_face_action == eNoFaceAction.SKIP_FRAME:
            #This only works with in-mem processing, as it simply skips the frame.
            #For 'extract frames' it simply leaves the unprocessed frame unprocessed and it gets used in the final output by ffmpeg.
            #If we could delete that frame here, that'd work but that might cause ffmpeg to fail unless the frames are renamed, and I don't think we have the info on what frame it actually is?????
            #alternatively, it could mark all the necessary frames for deletion, delete them at the end, then rename the remaining frames that might work?
            return None
        else:
            if num_swapped == NO_FACE_DETECTED and self.all_rotations_detected():
                # Multi-angle detection already looked at this frame at 0, 90,
                # 180 and 270 degrees and found nothing. The rotated retry would
                # run exactly those detections again (8 more passes) and return
                # the untouched frame.
                return frame
            return self.retry_rotated(frame)

    @staticmethod
    def all_rotations_detected():
        mode = unleashed.globals.multi_angle_detection_mode
        angles = set(unleashed.globals.multi_angle_angles or [])
        return mode in ('fallback', 'always') and {90, 180, 270} <= angles

    def retry_rotated(self, frame):
        # The rotated copies of a frame are not the video's frame sequence and
        # their coordinates are not the frame's: they are processed like a
        # single image -- no landmark smoothing, no target tracks, no
        # expression smoothing. They used to go into the stabilizer and the
        # tracks of the upright frames, out of frame order (in_order(None)) and
        # in rotated coordinates: each attempt aged every track and could take
        # over an upright face's track, so upright frames after a retry changed
        # and a render differed from run to run with several threads. The
        # faces a retry finds are lying ones, which the default path
        # (multi-angle + auto-rotate) also swaps with their own, unsmoothed
        # landmarks.
        tls = getattr(self, '_tls', None)
        frame_index = getattr(tls, 'frame_index', None)
        if tls is not None:
            tls.frame_index = None
            tls.rotated_retry = True
        try:
            return self._retry_rotated(frame)
        finally:
            if tls is not None:
                tls.frame_index = frame_index
                tls.rotated_retry = False

    def _retry_rotated(self, frame):
        copyframe = frame.copy()
        copyframe = rotate_clockwise(copyframe)
        temp_frame = copyframe.copy()
        num_swapped, temp_frame = self.swap_faces(copyframe, temp_frame)
        if num_swapped > 0:
            return rotate_anticlockwise(temp_frame)
        
        copyframe = frame.copy()
        copyframe = rotate_anticlockwise(copyframe)
        temp_frame = copyframe.copy()
        num_swapped, temp_frame = self.swap_faces(copyframe, temp_frame)
        if num_swapped > 0:
            return rotate_clockwise(temp_frame)
        del copyframe
        return frame
        


    def swap_faces(self, frame, temp_frame, frame_index=None):
        num_faces_found = 0

        # temporal smoothing is only safe for sequential video frames
        # a rotated retry of a frame (see retry_rotated) has none of these
        rotated_retry = bool(getattr(getattr(self, '_tls', None), 'rotated_retry', False))
        smoothing_on = (
            unleashed.globals.landmark_smoothing
            and (self.video_mode or unleashed.globals.force_landmark_smoothing)
            and self.stabilizer is not None
            and not rotated_retry
        )
        # identity / face-shape averages along each face's track: video only,
        # independent of landmark smoothing
        track_identity = float(getattr(unleashed.globals, 'identity_strength', 0.0) or 0.0) > 0.0
        track_shape = float(getattr(unleashed.globals, 'face_shape_strength', 0.0) or 0.0) > 0.0
        tracks_on = (self.video_mode and getattr(self, 'target_tracks', None) is not None
                     and (track_identity or track_shape) and not rotated_retry)
        mode = unleashed.globals.multi_angle_detection_mode
        angles = unleashed.globals.multi_angle_angles

        if self.options.swap_mode == "first":
            import time as _t
            _prof = getattr(unleashed.globals, 'profile_timings', False)
            _t0 = _t.perf_counter()
            face = get_first_face_multi(frame, mode=mode, angles=angles)
            _t1 = _t.perf_counter()

            if face is None:
                if _prof:
                    print(f"[timing] detect={ (_t1-_t0)*1000:.0f}ms  (no face)  frame={frame.shape[1]}x{frame.shape[0]}")
                self.sequencer.pass_turn(frame_index)
                return NO_FACE_DETECTED, frame

            # ordered: landmark refinement + temporal smoothing must see frames
            # in real sequence (the swap below stays parallel). Every video frame
            # takes its turn exactly once -- frames without a face too (above) --
            # and the order is kept whether or not smoothing is on, so switching
            # smoothing mid-render cannot desync the sequencer.
            with self.sequencer.in_order(frame_index):
                refine_faces_landmark68(frame, [face])
                if smoothing_on:
                    self.stabilizer.stabilize([face])
                if tracks_on:
                    self.target_tracks.update([face], identity=track_identity, shape=track_shape)

            num_faces_found += 1
            self._expect_faces(1)
            temp_frame = self.process_face(self.options.selected_index, face, temp_frame)
            if _prof:
                _t2 = _t.perf_counter()
                print(f"[timing] detect={(_t1-_t0)*1000:.0f}ms  swap+paste={(_t2-_t1)*1000:.0f}ms  "
                      f"frame={frame.shape[1]}x{frame.shape[0]}  subsample={unleashed.globals.subsample_size}")
            del face

        else:
            faces = get_all_faces_multi(frame, mode=mode, angles=angles)
            if faces is None:
                self.sequencer.pass_turn(frame_index)
                return NO_FACE_DETECTED, frame

            # ordered: see note above
            with self.sequencer.in_order(frame_index):
                refine_faces_landmark68(frame, faces)
                if smoothing_on:
                    self.stabilizer.stabilize(faces)
                if tracks_on:
                    self.target_tracks.update(faces, identity=track_identity, shape=track_shape)
            if self.options.swap_mode == "all":
                self._expect_faces(len(faces))
                for face in faces:
                    num_faces_found += 1
                    temp_frame = self.process_face(self.options.selected_index, face, temp_frame)

            elif self.options.swap_mode == "all_input" or self.options.swap_mode == "all_random":
                self._expect_faces(min(len(faces), len(self.input_face_datas)))
                for i,face in enumerate(faces):
                    num_faces_found += 1
                    if i < len(self.input_face_datas):
                        temp_frame = self.process_face(i, face, temp_frame)
                    else:
                        break
            
            elif self.options.swap_mode == "selected":
                # The picked source face ALWAYS applies here, exactly like in
                # "first" / "all" / "all_female" / "all_male".
                # Before: with 2+ marked target faces this used process_face(i,...)
                # -- pairing target i with INPUT face i -- and silently ignored the
                # source face picked in the UI. That is why changing the source did
                # nothing in "selected" mode while it worked in every other mode,
                # and why the same clip looked different between modes (a DIFFERENT
                # SOURCE FACE was applied, which reads as "different landmarking").
                # The 1:1 pairing behaviour still exists as its own mode: all_input.
                idx = self.options.selected_index
                if idx >= len(self.input_face_datas):
                    idx = 0
                if len(self.input_face_datas) > 0:
                    picked = match_selected_faces(self.target_face_datas, faces,
                                                  self.options.face_distance_threshold)
                    self._expect_faces(len(picked))
                    for fi in picked:
                        temp_frame = self.process_face(idx, faces[fi], temp_frame)
                        num_faces_found += 1
            elif self.options.swap_mode == "all_female" or self.options.swap_mode == "all_male":
                gender = 'F' if self.options.swap_mode == "all_female" else 'M'
                self._expect_faces(sum(1 for face in faces if face.sex == gender))
                for face in faces:
                    if face.sex == gender:
                        num_faces_found += 1
                        temp_frame = self.process_face(self.options.selected_index, face, temp_frame)
            
            # might be slower but way more clean to release everything here
            for face in faces:
                del face
            faces.clear()



        if unleashed.globals.vr_mode and num_faces_found % 2 > 0:
            # stereo image, there has to be an even number of faces
            num_faces_found = 0
            return num_faces_found, frame
        if num_faces_found == 0:
            return num_faces_found, frame

        #maskprocessor = next((x for x in self.processors if x.type == 'mask'), None)

        return num_faces_found, temp_frame


    def _expect_faces(self, count):
        """How many process_face calls this frame makes from here: the
        Expression Restorer's order gate opens after the last one's smoothing."""
        tls = getattr(self, '_tls', None)
        if tls is not None:
            tls.er_faces_left = count


    def set_manual_mask(self, painted):
        """Use this painting (editor layer or dict; None = none) from now on."""
        self.options.imagemask = self.manual_mask(painted)
        self._imagemask_fitted = None


    @staticmethod
    def manual_mask(imagemask):
        """The painted "keep the original here" mask: an editor dict or a layer
        array (RGBA: painted = alpha, RGB/gray: painted = bright). Returns a float
        mask in [0, 1] at the painting's own size, feathered, or None when
        nothing is painted."""
        layer = imagemask
        if isinstance(layer, dict):
            layers = layer.get("layers") or []
            layer = layers[0] if len(layers) > 0 else None
        if layer is None:
            return None
        layer = np.asarray(layer)
        if layer.ndim == 3 and layer.shape[2] == 4:
            painted = layer[:, :, 3]
        elif layer.ndim == 3:
            painted = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            painted = layer
        painted = np.where(painted > 0, 255, 0).astype(np.uint8)
        if not np.any(painted):
            return None
        k = max(3, int(round(max(painted.shape) / 200)) * 2 + 1)      # a soft edge, ~1% of the painting
        return cv2.GaussianBlur(painted, (k, k), 0).astype(np.float32) / 255


    def fitted_imagemask(self, frame):
        """The manual mask at this frame's size (painted on the downscaled
        preview, used on full-size frames), as H x W x 1."""
        h, w = frame.shape[:2]
        cached = getattr(self, '_imagemask_fitted', None)
        if cached is None or cached.shape[:2] != (h, w):
            m = self.options.imagemask
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            cached = m[:, :, None]
            self._imagemask_fitted = cached
        return cached


    def rotation_action(self, original_face:Face, frame:Frame):
        (height, width) = frame.shape[:2]

        bounding_box_width = original_face.bbox[2] - original_face.bbox[0]
        bounding_box_height = original_face.bbox[3] - original_face.bbox[1]
        horizontal_face = bounding_box_width > bounding_box_height

        center_x = width // 2.0
        start_x = original_face.bbox[0]
        end_x = original_face.bbox[2]
        bbox_center_x = start_x + (bounding_box_width // 2.0)

        # need to leverage the array of landmarks as decribed here:
        # https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg
        # basically, we should be able to check for the relative position of eyes and nose
        # then use that to determine which way the face is actually facing when in a horizontal position
        # and use that to determine the correct rotation_action

        forehead_x = original_face.landmark_2d_106[72][0]
        chin_x = original_face.landmark_2d_106[0][0]

        if horizontal_face:
            if chin_x < forehead_x:
                # this is someone lying down with their face like this (:
                return "rotate_anticlockwise"
            elif forehead_x < chin_x:
                # this is someone lying down with their face like this :)
                return "rotate_clockwise"
            if bbox_center_x >= center_x:
                # this is someone lying down with their face in the right hand side of the frame
                return "rotate_anticlockwise"
            if bbox_center_x < center_x:
                # this is someone lying down with their face in the left hand side of the frame
                return "rotate_clockwise"

        return None


    def auto_unrotate_frame(self, frame:Frame, rotation_action):
        if rotation_action == "rotate_anticlockwise":
            return rotate_clockwise(frame)
        elif rotation_action == "rotate_clockwise":
            return rotate_anticlockwise(frame)
        
        return frame



    def process_face(self,face_index, target_face:Face, frame:Frame):
        from unleashed.face_util import align_crop, align_crop_robust, landmark_68_to_5

        enhanced_frame = None
        if(len(self.input_face_datas) > 0):
            inputface = self.input_face_datas[face_index].faces[0]
        else:
            inputface = None

        rotation_action = None
        if unleashed.globals.autorotate_faces:
            # check for sideways rotation of face
            rotation_action = self.rotation_action(target_face, frame)
            if rotation_action is not None:
                (startX, startY, endX, endY) = target_face["bbox"].astype("int")
                width = endX - startX
                height = endY - startY
                offs = int(max(width,height) * 0.25)
                rotcutframe,startX, startY, endX, endY = self.cutout(frame, startX - offs, startY - offs, endX + offs, endY + offs)
                if rotation_action == "rotate_anticlockwise":
                    rotcutframe = rotate_anticlockwise(rotcutframe)
                elif rotation_action == "rotate_clockwise":
                    rotcutframe = rotate_clockwise(rotcutframe)
                # rotate image and re-detect face to correct wonky landmarks
                rotface = get_first_face(rotcutframe)
                if rotface is None:
                    rotation_action = None
                else:
                    # Run the same landmark refinement as the normal path (the
                    # rotated retry used to skip it, so exactly the hardest
                    # frames -- horizontal faces -- were aligned with the rawest
                    # landmarks). The temporal stabilizer is deliberately NOT
                    # applied here: rotcutframe coordinates live in a different
                    # space than the full frame, so its tracks would mismatch.
                    refine_faces_landmark68(rotcutframe, [rotface])
                    saved_frame = frame.copy()
                    frame = rotcutframe
                    target_face = rotface



        # if unleashed.globals.vr_mode:
            # bbox = target_face.bbox
            # [orig_width, orig_height, _] = frame.shape

            # # Convert bounding box to ints
            # x1, y1, x2, y2 = map(int, bbox)

            # # Determine the center of the bounding box
            # x_center = (x1 + x2) / 2
            # y_center = (y1 + y2) / 2

            # # Normalize coordinates to range [-1, 1]
            # x_center_normalized = x_center / (orig_width / 2) - 1
            # y_center_normalized = y_center / (orig_width / 2) - 1

            # # Convert normalized coordinates to spherical (theta, phi)
            # theta = x_center_normalized * 180  # Theta ranges from -180 to 180 degrees
            # phi = -y_center_normalized * 90  # Phi ranges from -90 to 90 degrees

            # img = vr.GetPerspective(frame, 90, theta, phi, 1280, 1280)  # Generate perspective image


        """ Code ported/adapted from Facefusion which borrowed the idea from Rope:
            Kind of subsampling the cutout and aligned face image and faceswapping slices of it up to
            the desired output resolution. This works around the current resolution limitations without using enhancers.
        """
        # Face shape from the source (off at 0): the frame around this face is
        # warped so the target's jaw / chin take the source's shape, BEFORE the
        # crop is taken; inswapper keeps the shape it is given, and the face is
        # pasted onto the warped frame. Brows, eyes, nose and lips do not move,
        # so the alignment is unchanged. Not for auto-rotated faces (their
        # landmarks live in the rotated cut-out). See unleashed/face_shape.py.
        shape_strength = float(getattr(unleashed.globals, 'face_shape_strength', 0.0) or 0.0)
        if shape_strength > 0.0 and inputface is not None and rotation_action is None:
            try:
                from unleashed.face_shape import warp_frame
                frame = warp_frame(frame, target_face, self.input_face_datas[face_index], shape_strength)
            except Exception as e:
                if not getattr(self, '_shape_error_logged', False):
                    self._shape_error_logged = True
                    print(f'[face-shape] skipped: {e}')

        model_output_size = self.options.swap_output_size
        subsample_size = max(self.options.subsample_size, model_output_size)
        subsample_total = subsample_size // model_output_size

        # Alignment: derive 5 stable keypoints from the 68-point landmark model
        # and fit a RANSAC affine. This keeps inswapper accurate at extreme
        # yaw/pitch where the detector's raw kps get noisy. Falls back cleanly
        # to the original kps + similarity transform if anything is missing.
        landmarks68 = getattr(target_face, "landmark_3d_68", None)
        if unleashed.globals.use_landmark_alignment and landmarks68 is not None:
            try:
                lmk5 = landmark_68_to_5(landmarks68)
                # Landmark sanity gate -- OPT-IN. When the toggle is off this
                # whole block is skipped, so nothing is computed and no frame can
                # ever change alignment basis (zero cost, zero behaviour).
                #
                # What it does when on: the 68->5 points and the detector's own 5
                # kps should mark the same eyes/nose/mouth. A large disagreement
                # means the 68pt model broke for this frame (extreme pose, blur,
                # occlusion), so we fall back to the kps for that frame.
                #
                # Know the trade-off: the two point sets have a SYSTEMATIC offset
                # (different models, different definitions), so a threshold that
                # is too low makes the gate fire on ordinary frames and the
                # alignment flips between two bases that sit measurably apart --
                # which is itself a flicker source. Higher = fires only on real
                # breakage. Set the threshold above your footage's normal
                # disagreement, or leave the gate off entirely.
                if getattr(unleashed.globals, 'landmark_sanity_gate', False):
                    kps = getattr(target_face, 'kps', None)
                    if kps is not None:
                        kps5 = np.asarray(kps, dtype=np.float32).reshape(-1, 2)[:5]
                        if kps5.shape == lmk5.shape:
                            bb = np.asarray(target_face.bbox, dtype=np.float32)
                            fsize = float(max(bb[2] - bb[0], bb[3] - bb[1])) + 1e-6
                            d = np.linalg.norm(lmk5 - kps5, axis=1) / fsize
                            d_mean = float(d.mean())
                            d_max = float(d.max())
                            thr = float(getattr(unleashed.globals, 'landmark_sanity_threshold', 0.20))
                            # mean catches a globally-drifted landmark set; the
                            # per-point max (2x thr) catches the commoner failure
                            # where ONE point goes wild and the mean dilutes it.
                            if d_mean > thr or d_max > 2.0 * thr:
                                if getattr(unleashed.globals, 'expression_debug', False):
                                    print(f"[lmk-gate] disagree mean={d_mean:.3f} max={d_max:.3f} "
                                          f"(thr {thr:.3f}/{2*thr:.3f}, face {fsize:.0f}px) -> kps alignment")
                                raise ValueError('landmark sanity gate tripped')
                aligned_img, M = align_crop_robust(frame, lmk5, subsample_size)
            except Exception:
                # Gate tripped, or a hard failure (missing / != 68 landmarks).
                aligned_img, M = align_crop(frame, target_face.kps, subsample_size)
        else:
            aligned_img, M = align_crop(frame, target_face.kps, subsample_size)

        fake_frame = aligned_img
        if self.options.show_face_masking:
            # Mask view (Face Swap tab): no swap processor runs; the "swapped face"
            # is the crop tinted green, so after the occlusion mask, the paste
            # matte (erosion, blur, offsets) and the kept mouth / eyes, the frame
            # is tinted exactly where a swap would replace it.
            fake_frame = cv2.addWeighted(aligned_img, 0.45, np.full_like(aligned_img, (0, 255, 0)), 0.55, 0)
        target_face.matrix = M

        import time as _pt
        _prof = getattr(unleashed.globals, 'profile_timings', False)
        for p in self.processors:
            _ps = _pt.perf_counter()
            if p.type == 'swap':
                swap_result_frames = []
                subsample_frames = self.implode_pixel_boost(aligned_img, model_output_size, subsample_total)
                for sliced_frame in subsample_frames:
                    for _ in range(0,self.options.num_swap_steps):
                        sliced_frame = self.prepare_crop_frame(sliced_frame)
                        sliced_frame = p.Run(inputface, target_face, sliced_frame)
                        sliced_frame = self.normalize_swap_frame(sliced_frame)
                    swap_result_frames.append(sliced_frame)
                fake_frame = self.explode_pixel_boost(swap_result_frames, model_output_size, subsample_total, subsample_size)
                fake_frame = fake_frame.astype(np.uint8)
                scale_factor = 0.0
            elif p.type == 'mask':
                # Run the mask model ONCE and re-apply the same raw mask to both
                # frames (the old code ran a full ONNX inference twice on the
                # identical aligned crop when the enhanced face also needed it).
                raw_mask = self.compute_mask(p, aligned_img)
                fake_frame = self.apply_mask(raw_mask, aligned_img, fake_frame)
                # When the mask runs after the enhancer, the enhanced face already
                # exists and is the dominant source in paste_upscale's blend, so
                # the occluder must be restored onto it too -- otherwise it would
                # be diluted/lost. (Before the enhancer, enhanced_frame is None.)
                if enhanced_frame is not None:
                    enhanced_frame = self.apply_mask(raw_mask, aligned_img, enhanced_frame)
            elif p.type == 'expression':
                # Guard the ER exactly like the enhancer: a failure here must NOT
                # abort the frame and skip the occlusion mask (which runs after).
                # On failure we keep the un-restored swapped face and carry on so
                # the mask still restores occluders (glass/hand/hair).
                try:
                    fake_frame = self.process_expression(p, aligned_img, fake_frame, frame, target_face)
                except Exception as _e:
                    if getattr(unleashed.globals, 'expression_debug', False):
                        print(f"[expression] '{p.processorname}' failed on a frame ({_e}); using un-restored face")
            else:
                # Enhancer pass. Guard against a failed/garbage result (e.g. a
                # transient CUDA OOM on a conv -> NaN -> black crop) which would
                # paste as a raw rectangular box around the head. On failure we
                # keep enhanced_frame = None so the un-enhanced face is pasted
                # instead (a slightly softer single frame, far less visible).
                try:
                    _ef, _sf = p.Run(self.input_face_datas[face_index], target_face, fake_frame)
                    if _ef is not None and float(np.asarray(_ef).max()) > 1:
                        enhanced_frame, scale_factor = _ef, _sf
                    elif getattr(unleashed.globals, 'expression_debug', False):
                        print(f"[enhancer] '{p.processorname}' returned an empty/black frame; using un-enhanced")
                except Exception as _e:
                    if getattr(unleashed.globals, 'expression_debug', False):
                        print(f"[enhancer] '{p.processorname}' failed on a frame ({_e}); using un-enhanced")
            if _prof:
                print(f"[timing]    processor '{p.processorname}' ({p.type}) = {(_pt.perf_counter()-_ps)*1000:.0f}ms")

        # Paste-back working size. 512 is the floor: a small swap (128/256) is
        # upscaled first so the warp back into the frame is smooth instead of
        # blocky. It must NOT be a ceiling -- with a hard 512 a 768/1024 subsample
        # was resized back DOWN to 512, throwing away exactly the resolution it
        # was computed for.
        orig_width = fake_frame.shape[1]
        upscale = max(512, orig_width)
        if orig_width != upscale:
            fake_frame = cv2.resize(fake_frame, (upscale, upscale), interpolation=cv2.INTER_CUBIC)
        mask_offsets = (0,0,0,0,1,20) if inputface is None else inputface.mask_offsets

        
        if enhanced_frame is None:
            # float, not int: this is the ratio between the paste-back size and
            # the swap size. int() truncated it to 0 for any subsample larger
            # than `upscale` (768 -> int(512/768) = 0), and paste_upscale does
            # `M_scale = M * scale_factor`, so the whole affine matrix became
            # zeros -> invertAffineTransform produced a degenerate transform that
            # smeared the paste across the entire frame == the black screen.
            scale_factor = float(upscale) / float(orig_width)
            result = self.paste_upscale(fake_frame, fake_frame, target_face.matrix, frame, scale_factor, mask_offsets, target_face)
        else:
            result = self.paste_upscale(fake_frame, enhanced_frame, target_face.matrix, frame, scale_factor, mask_offsets, target_face)

        # Restore mouth before unrotating
        if self.options.restore_original_mouth:
            mouth_cutout, mouth_bb = self.create_mouth_mask(target_face, frame)
            result = self.apply_mouth_area(result, mouth_cutout, mouth_bb)

        # Restore original eyes (inside-eye region only; brows/lids excluded).
        if getattr(self.options, 'restore_original_eyes', False):
            eyes = self.create_eyes_masks(target_face, frame)
            result = self.apply_eyes_area(result, eyes)

        if rotation_action is not None:
            fake_frame = self.auto_unrotate_frame(result, rotation_action)
            result = self.paste_simple(fake_frame, saved_frame, startX, startY)
        
        return result

        


    def cutout(self, frame:Frame, start_x, start_y, end_x, end_y):
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if end_x > frame.shape[1]:
            end_x = frame.shape[1]
        if end_y > frame.shape[0]:
            end_y = frame.shape[0]
        return frame[start_y:end_y, start_x:end_x], start_x, start_y, end_x, end_y

    def paste_simple(self, src:Frame, dest:Frame, start_x, start_y):
        end_x = start_x + src.shape[1]
        end_y = start_y + src.shape[0]

        start_x, end_x, start_y, end_y = clamp_cut_values(start_x, end_x, start_y, end_y, dest)
        dest[start_y:end_y, start_x:end_x] = src
        return dest
        
    def simple_blend_with_mask(self, image1, image2, mask):
        # Blend the images
        blended_image = image1.astype(np.float32) * (1.0 - mask) + image2.astype(np.float32) * mask
        return blended_image.astype(np.uint8)


    def paste_upscale(self, fake_face, upsk_face, M, target_img, scale_factor, mask_offsets, target_face=None):
        M_scale = M * scale_factor
        IM = cv2.invertAffineTransform(M_scale)

        # Generate white square sized as a upsk_face
        img_matte = np.zeros((upsk_face.shape[0],upsk_face.shape[1]), dtype=np.uint8)

        w = img_matte.shape[1]
        h = img_matte.shape[0]

        top = int(mask_offsets[0] * h)
        bottom = int(h - (mask_offsets[1] * h))
        left = int(mask_offsets[2] * w)
        right = int(w - (mask_offsets[3] * w))
        img_matte[top:bottom,left:right] = 255

        # erosion / feather are global settings now (see unleashed/globals.py)
        erosion = int(getattr(unleashed.globals, 'mask_erosion_iterations', 1))
        blur = int(getattr(unleashed.globals, 'mask_blur_size', 20))

        # Only the paste area is processed. The matte is zero outside the warped
        # face square, so eroding, blurring, warping and blending the whole frame
        # was wasted work (~0.13 s per face at 1080p, ~0.6 s at 4K, on CPU).
        area = self.paste_area(IM, w, h, target_img.shape, erosion, blur)
        if area is None:
            # the face square lies outside the frame: nothing to paste (the
            # overlay still dims the frame, as it always did)
            result = target_img.copy()
            if self.options.show_face_area_overlay:
                result = cv2.addWeighted(result, 1 - 0.5, np.zeros_like(target_img), 0.5, 0)
            return result
        x0, y0, x1, y1 = area
        target_area = target_img[y0:y1, x0:x1]

        # Transform white area back to target_img (INTER_LINEAR for soft,
        # anti-aliased edges instead of the stair-stepped INTER_NEAREST).
        # The warps start at the frame origin (dsize x1,y1) and are cropped
        # afterwards: warpAffine's fixed-point coordinates depend on the
        # absolute pixel position, so a warp with the matrix shifted to x0,y0
        # rounds differently (OpenCV 4.10: up to 11 levels). Starting at the
        # origin keeps every pixel identical to the old full-frame warp.
        square = img_matte
        face_aligned = getattr(unleashed.globals, 'mask_face_aligned', False)
        if not face_aligned:
            img_matte = cv2.warpAffine(square, IM, (x1, y1), flags=cv2.INTER_LINEAR, borderValue=0.0)[y0:y1, x0:x1]

        # NOTE: the frame border used to be zeroed HERE, before blur_area. That
        # was wrong: blur_area erodes with a kernel scaled to the face size, so
        # the freshly-created zero along the image edge got eaten inward and the
        # swap stopped 10-20px short of the picture edge (a visible "inner
        # frame" whenever a face sits against the side of the shot). blur_area
        # now replicates the border while eroding, so a matte that legitimately
        # touches the frame edge keeps reaching it. On the paste area this is
        # exact: its sides inside the frame lie beyond the matte's reach (the
        # matte is zero there), its sides on the frame edge are the frame edge.
        to_chin = getattr(unleashed.globals, 'mask_bottom_to_chin', False)
        if face_aligned:
            img_matte = self.face_aligned_matte(IM, area, square.shape, (top, bottom, left, right), erosion, blur, to_chin)
        elif to_chin:
            img_matte = self.matte_to_chin(img_matte, square, IM, area, (top, bottom, left, right), erosion, blur)
        else:
            img_matte = self.blur_area(img_matte, erosion, blur)
            #Normalize images to float values and reshape
            img_matte = img_matte.astype(np.float32)/255
        if self.options.show_face_area_overlay:
            # Additional steps for green overlay
            green_overlay = np.zeros_like(target_img)
            green_area = green_overlay[y0:y1, x0:x1]
            green_color = [0, 255, 0]  # RGB for green
            for i in range(3):  # Apply green color where img_matte is not zero
                green_area[:, :, i] = np.where(img_matte > 0, green_color[i], 0)        ##Transform upcaled face back to target_img
        img_matte = np.reshape(img_matte, [img_matte.shape[0],img_matte.shape[1],1])
        paste_face = cv2.warpAffine(upsk_face, IM, (x1, y1), borderMode=cv2.BORDER_REPLICATE)
        if upsk_face is not fake_face:
            # IM is calibrated for upsk_face's size. With an enhancer the enhanced
            # face is 512 (all three enhancer models are 512-in/512-out) while the
            # un-enhanced swap can be 768/1024, and warping a differently-sized
            # image with the same matrix lands it zoomed and misaligned. Match the
            # sizes before warping so the blend actually overlaps.
            if fake_face.shape[:2] != upsk_face.shape[:2]:
                fake_face = cv2.resize(fake_face, (upsk_face.shape[1], upsk_face.shape[0]),
                                       interpolation=cv2.INTER_AREA)
            fake_face = cv2.warpAffine(fake_face, IM, (x1, y1), borderMode=cv2.BORDER_REPLICATE)
            # blended BEFORE cropping to the paste area: addWeighted rounds the
            # last elements of a row differently (scalar vs SIMD), so blending
            # the cropped views changed a pixel by 1 level now and then
            paste_face = cv2.addWeighted(paste_face, self.options.blend_ratio, fake_face, 1.0 - self.options.blend_ratio, 0)
        paste_face = paste_face[y0:y1, x0:x1]

        # Optional Reinhard LAB color transfer toward the target, restricted to
        # the face region so background lighting does not skew the statistics.
        if unleashed.globals.use_color_transfer and not self.options.show_face_masking:
            paste_face = self._match_color_masked(paste_face, target_area, img_matte)

        # Re-assemble image (outside the paste area the matte is zero, so the
        # frame is left exactly as it was)
        paste_face = img_matte * paste_face
        paste_face = paste_face + (1-img_matte) * target_area.astype(np.float32)
        result = target_img.copy()
        result[y0:y1, x0:x1] = paste_face.astype(np.uint8)
        if self.options.show_face_area_overlay:
            # Overlay the green overlay on the final image
            result = cv2.addWeighted(result, 1 - 0.5, green_overlay, 0.5, 0)
        return result


    def face_aligned_matte(self, IM, area, shape, box, erosion, blur, to_chin):
        """The paste matte built in the face crop's own coordinates, then warped
        to the frame. blur_area erodes and feathers in FRAME pixels with an
        axis-aligned square kernel sized from the matte's bounding box, so a
        tilted face square is eroded deeper and feathered wider than an upright
        one (about 75% deeper at 25 degrees): tilted faces kept more of the
        target at the forehead, cheeks and chin. Here every tilt gets what an
        upright face gets. to_chin: the bottom edge is not eroded and fades out
        over the last 6% of the crop (see matte_to_chin). Returns a float matte
        in [0, 1] on the paste area."""
        x0, y0, x1, y1 = area
        h, w = shape[:2]
        top, bottom, left, right = box
        # frame pixels per crop pixel. The canvas is drawn at r canvas pixels per
        # crop pixel: at frame resolution for a face smaller in the frame than
        # the crop (no point in a finer matte), at crop resolution otherwise.
        scale = float(np.sqrt(abs(np.linalg.det(IM[:, :2])))) or 1.0
        r = min(1.0, scale)
        # blur_area's kernels for the same face upright (sized from the white
        # area in frame pixels), in canvas pixels
        size = int(np.sqrt(max(bottom - top, 1) * max(right - left, 1)) * scale)
        k = max(1, int(round(max(size // 10, 10) * r / scale)))
        k2 = max(1, int(round(max((size * blur) // 400, 4) * r / scale)))
        pad = k * max(1, erosion) + 2 * k2 + 4
        ch, cw = int(round(h * r)), int(round(w * r))
        t, b, l, rt = (int(round(v * r)) for v in (top, bottom, left, right))
        canvas = np.zeros((ch + 2 * pad, cw + 2 * pad), dtype=np.uint8)
        white_bottom = canvas.shape[0] if to_chin else pad + b
        canvas[pad + t:white_bottom, pad + l:pad + rt] = 255
        canvas = cv2.erode(canvas, np.ones((k, k), np.uint8), iterations=erosion)
        canvas = cv2.GaussianBlur(canvas, (2 * k2 + 1, 2 * k2 + 1), 0).astype(np.float32) / 255
        if to_chin:
            fade = max(2.0, 0.06 * h * r)
            ys = np.arange(canvas.shape[0], dtype=np.float32) - pad
            rows = np.clip((b - 1 - ys) / fade, 0.0, 1.0)
            canvas *= (rows * rows * (3.0 - 2.0 * rows))[:, None]
        # canvas pixel c -> crop pixel (c - pad) / r -> frame via IM -> paste area
        A = IM[:, :2] / r
        IM_canvas = np.hstack([A, (IM[:, 2] - A @ np.array([pad, pad], dtype=np.float64)
                                   - np.array([x0, y0], dtype=np.float64))[:, None]])
        return cv2.warpAffine(canvas, IM_canvas, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0.0)


    def matte_to_chin(self, img_matte, square, IM, area, box, erosion, blur):
        """The paste matte with the 'Extend swap to the chin' option. The top and
        side edges get exactly their usual erosion + feather; the bottom edge
        instead fades out over the last 6% of the crop. The arcface crop puts the
        chin at ~87-92% of its height, so the usual erosion + feather band of the
        bottom edge landed on the chin and left the target's chin unswapped.
        img_matte is the warped square on the paste area; returns a float matte
        in [0, 1]."""
        x0, y0, x1, y1 = area
        h, w = square.shape[:2]
        top, bottom, left, right = box
        mask_size = self.matte_size(img_matte)
        if mask_size is None:
            return img_matte.astype(np.float32) / 255
        # the square continued well below the crop, so erosion + feather only
        # shape its top and sides -- with the kernels of the real square
        tall = np.zeros((h + h // 2, w), dtype=np.uint8)
        tall[top:, left:right] = 255
        tall = cv2.warpAffine(tall, IM, (x1, y1), flags=cv2.INTER_LINEAR, borderValue=0.0)[y0:y1, x0:x1]
        tall = self.blur_area(tall, erosion, blur, mask_size=mask_size).astype(np.float32) / 255
        # the bottom edge: a smooth fade over the last 6% above `bottom`, zero
        # from there down (below the crop there are no swapped pixels). The fade
        # is 1 beyond the other sides, leaving those edges to the matte above.
        fade = max(2, int(round(0.06 * h)))
        rows = np.clip((bottom - 1 - np.arange(h, dtype=np.float32)) / fade, 0.0, 1.0)
        rows = rows * rows * (3.0 - 2.0 * rows)
        edge = np.repeat(rows[:, None], w, axis=1)
        edge = cv2.warpAffine(edge, IM, (x1, y1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[y0:y1, x0:x1]
        return np.minimum(tall, edge)


    @staticmethod
    def paste_area(IM, w, h, frame_shape, erosion, blur):
        """Frame rectangle (x0, y0, x1, y1) that paste_upscale has to touch: the
        w x h face square warped by IM, widened by the furthest that
        blur_area's erosion + feather can reach (its kernel sizes, computed
        from an upper bound of the square's size). None when the square lies
        entirely outside the frame."""
        frame_h, frame_w = frame_shape[:2]
        corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32)
        quad = cv2.transform(corners, IM)[0]
        qx0, qy0 = np.floor(quad.min(axis=0)).astype(int) - 1
        qx1, qy1 = np.ceil(quad.max(axis=0)).astype(int) + 1
        size = int(np.sqrt(max(qx1 - qx0, 1) * max(qy1 - qy0, 1)))
        k = max(size // 10, 10)
        k2 = max((size * blur) // 400, 4)
        margin = int(k * max(1, erosion) + 2 * k2 + 4)
        x0, y0 = max(int(qx0) - margin, 0), max(int(qy0) - margin, 0)
        x1, y1 = min(int(qx1) + margin, frame_w), min(int(qy1) + margin, frame_h)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1


    def _match_color_masked(self, src, dst, mask):
        """Reinhard color transfer (LAB mean/std) from dst->src over the masked
        face region only. src/dst are uint8 BGR full-frame images, mask is a
        float HxWx1 in [0,1]."""
        try:
            m = (mask[:, :, 0] > 0.1)
            if m.sum() < 32:
                return src
            src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
            dst_lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype(np.float32)
            for c in range(3):
                s = src_lab[:, :, c][m]
                d = dst_lab[:, :, c][m]
                s_mean, s_std = s.mean(), s.std() + 1e-6
                d_mean, d_std = d.mean(), d.std() + 1e-6
                src_lab[:, :, c] = (src_lab[:, :, c] - s_mean) * (d_std / s_std) + d_mean
            src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
            return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)
        except Exception:
            return src
        # (an unreachable duplicated copy of paste_upscale's tail used to live
        # here after the return -- removed.)


    @staticmethod
    def matte_size(img_matte):
        """Size (geometric mean of height and width) of the white area of a
        matte, which scales blur_area's kernels. None for an empty matte."""
        # Detect the affine transformed white area
        mask_h_inds, mask_w_inds = np.where(img_matte==255)
        if mask_h_inds.size == 0:
            # No pixel is exactly 255 (e.g. an anti-aliased user image mask):
            # fall back to the non-zero area; if the matte is fully empty there
            # is nothing to feather (instead of crashing on np.max([])).
            mask_h_inds, mask_w_inds = np.where(img_matte > 0)
            if mask_h_inds.size == 0:
                return None
        # Calculate the size (and diagonal size) of transformed white area width and height boundaries
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        return int(np.sqrt(mask_h*mask_w))


    def blur_area(self, img_matte, num_erosion_iterations, blur_amount, mask_size=None):
        # mask_size: pass the size of another matte to get exactly its kernels
        if mask_size is None:
            mask_size = self.matte_size(img_matte)
            if mask_size is None:
                return img_matte
        # Calculate the kernel size for eroding img_matte by kernel (insightface empirical guess for best size was max(mask_size//10,10))
        # k = max(mask_size//12, 8)
        # Erosion and feather used to share ONE number, and inverted at that:
        # k(erosion) = mask_size//(blur_amount//2) and k(blur) = mask_size//blur_amount,
        # so raising "Blur size" SHRANK the feather (30->15->10 px) while lowering
        # it doubled the erosion (30->60 px). Wanting a softer seam without losing
        # swap area was therefore impossible. They are decoupled now:
        #   erosion  -> fixed base kernel + the separate Erosion (iterations) slider
        #   feather  -> scales with "Blur size", monotonically
        # blur_amount == 20 (the default) reproduces the old numbers exactly.
        k = max(mask_size // 10, 10)
        kernel = np.ones((k,k),np.uint8)
        # Pad with BORDER_REPLICATE before eroding/blurring. Without it the area
        # outside the image counts as background, so a face touching the edge of
        # the frame is eroded inward from that edge as well -- the swap then
        # fades out before the picture border. Replicating keeps the edge value,
        # so only REAL mask borders are feathered. The pad is removed at the end.
        k2 = max((mask_size * int(blur_amount)) // 400, 4)
        pad = int(k * max(1, num_erosion_iterations) + 2 * k2 + 4)
        img_matte = cv2.copyMakeBorder(img_matte, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        img_matte = cv2.erode(img_matte,kernel,iterations = num_erosion_iterations)
        #Calculate the kernel size for blurring img_matte by blur_size (insightface empirical guess for best size was max(mask_size//20, 5))
        # k = max(mask_size//24, 4) 
        kernel_size = (k2, k2)
        blur_size = tuple(2*i+1 for i in kernel_size)
        img_matte = cv2.GaussianBlur(img_matte, blur_size, 0)
        # drop the replication pad -> back to the original frame size
        h_p, w_p = img_matte.shape[:2]
        img_matte = img_matte[pad:h_p - pad, pad:w_p - pad]
        # Now that feathering is done, kill exactly one pixel along the image
        # border so the matte can never bleed outside the frame during the
        # float blend. One pixel is invisible, unlike the old pre-erosion zero.
        img_matte[:1, :] = img_matte[-1:, :] = img_matte[:, :1] = img_matte[:, -1:] = 0
        return img_matte


    def prepare_crop_frame(self, swap_frame):
        model_type = 'inswapper'
        model_mean = [0.0, 0.0, 0.0]
        model_standard_deviation = [1.0, 1.0, 1.0]

        if model_type == 'ghost':
            swap_frame = swap_frame[:, :, ::-1] / 127.5 - 1
        else:
            swap_frame = swap_frame[:, :, ::-1] / 255.0
        swap_frame = (swap_frame - model_mean) / model_standard_deviation
        swap_frame = swap_frame.transpose(2, 0, 1)
        swap_frame = np.expand_dims(swap_frame, axis = 0).astype(np.float32)
        return swap_frame


    def normalize_swap_frame(self, swap_frame):
        model_type = 'inswapper'
        swap_frame = swap_frame.transpose(1, 2, 0)

        if model_type == 'ghost':
            swap_frame = (swap_frame * 127.5 + 127.5).round()
        else:
            swap_frame = (swap_frame * 255.0).round()
        # inswapper (a GAN) can over/undershoot slightly outside [0,255]. Without
        # clipping, the later .astype(uint8) WRAPS those values (e.g. -3 -> 253,
        # 258 -> 2), peppering dark cavities (nostrils, mouth) with wrong pixels.
        swap_frame = np.clip(swap_frame, 0.0, 255.0)
        swap_frame = swap_frame[:, :, ::-1]
        return swap_frame

    def implode_pixel_boost(self, aligned_face_frame, model_size, pixel_boost_total : int):
        subsample_frame = aligned_face_frame.reshape(model_size, pixel_boost_total, model_size, pixel_boost_total, 3)
        subsample_frame = subsample_frame.transpose(1, 3, 0, 2, 4).reshape(pixel_boost_total ** 2, model_size, model_size, 3)
        return subsample_frame


    def explode_pixel_boost(self, subsample_frame, model_size, pixel_boost_total, pixel_boost_size):
        final_frame = np.stack(subsample_frame, axis = 0).reshape(pixel_boost_total, pixel_boost_total, model_size, model_size, 3)
        final_frame = final_frame.transpose(2, 0, 3, 1, 4).reshape(pixel_boost_size, pixel_boost_size, 3)
        return final_frame

    def process_expression(self, processor, aligned_img:Frame, fake_frame:Frame,
                           frame:Frame=None, target_face=None):
        # aligned_img = original target crop (real expression -> driving)
        # fake_frame  = swapped crop (appearance/identity to keep)
        # Plain division, NOT np.interp: interp clamps at the ends, so with a
        # 0-300 slider everything above 100 would silently map to 1.0.
        factor = max(0.0, float(unleashed.globals.expression_restorer_factor) / 100.0)
        context = None
        last_index = None
        if frame is not None and target_face is not None:
            lmk = getattr(target_face, 'landmark_2d_106', None)
            if lmk is None:
                lmk = getattr(target_face, 'landmark_3d_68', None)
            context = {
                'frame': frame,
                'landmarks': lmk,
                'M': getattr(target_face, 'matrix', None),
                'pose': getattr(target_face, 'pose', None),
            }
            # temporal smoothing of the expression: consecutive frames of an
            # in-memory video only (not previews, images or extracted frames),
            # and in frame order (the gate), like the landmark smoothing
            frame_index = getattr(self._tls, 'frame_index', None)
            if (self.video_mode and frame_index is not None and getattr(self, 'exp_smoother', None) is not None
                    and float(getattr(unleashed.globals, 'expression_smoothing_strength', 0.0) or 0.0) > 0.0):
                context['exp_smoother'] = self.exp_smoother
                context['exp_order_wait'] = lambda i=frame_index: self.er_order.wait(i)
                # the frame's last face: the next frames may smooth as soon as
                # this one has (the fallback is the end of the frame)
                left = getattr(self._tls, 'er_faces_left', None)
                if left is not None:
                    left -= 1
                    self._tls.er_faces_left = left
                    if left <= 0:
                        last_index = frame_index
                        context['exp_order_done'] = lambda i=frame_index: self.er_order.done(i)
        try:
            restored = processor.Run(
                aligned_img, fake_frame, factor,
                unleashed.globals.expression_restore_eyes,
                unleashed.globals.expression_restore_mouth,
                unleashed.globals.expression_restore_brows,
                context,
            )
        finally:
            if last_index is not None:          # no smoothing step (pose gate) or an error
                self.er_order.done(last_index)
        return restored

    def compute_mask(self, processor, frame:Frame):
        """One mask-model inference on the aligned crop; returns the raw mask
        (model resolution). Split out so the result can be applied to several
        targets (swapped + enhanced) without re-running the model."""
        return processor.Run(frame, self.options.masking_text)

    def apply_mask(self, raw_mask, frame:Frame, target:Frame):
        img_mask = cv2.resize(raw_mask, (target.shape[1], target.shape[0]))
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])

        # The restore-source (frame = original aligned crop) must match the target
        # resolution before blending. They differ when masking the enhanced face
        # (e.g. 512) while the aligned crop is smaller (e.g. 256) -- this happens
        # when the occlusion mask runs after the enhancer.
        if frame.shape[:2] != target.shape[:2]:
            frame = cv2.resize(frame, (target.shape[1], target.shape[0]))

        target = target.astype(np.float32)
        result = (1-img_mask) * target
        result += img_mask * frame.astype(np.float32)
        return np.uint8(result)

    def process_mask(self, processor, frame:Frame, target:Frame):
        # kept for compatibility with any external callers
        return self.apply_mask(self.compute_mask(processor, frame), frame, target)


    # Mouth restoration: adapted from iVideoGameBoss's DeepFaceCam (see README.md, Credits)
    
    def create_mouth_mask(self, face: Face, frame: Frame):
        mouth_cutout = None
        
        landmarks = face.landmark_2d_106
        if landmarks is not None:
            # Get mouth landmarks (indices 52 to 71 typically represent the outer mouth)
            mouth_points = landmarks[52:71].astype(np.int32)
            
            min_x, min_y = np.min(mouth_points, axis=0)
            max_x, max_y = np.max(mouth_points, axis=0)

            # Pad proportionally to the mouth size rather than with fixed pixel
            # amounts. The original used absolute paddings (up to +540px below the
            # mouth) which, on small faces, grabbed the whole chin/neck and hurt
            # both identity and expression restoration. Proportional padding keeps
            # the region tight to the mouth at any face scale.
            mouth_w = max(1, int(max_x - min_x))
            mouth_h = max(1, int(max_y - min_y))
            pad_x = int(mouth_w * 0.6)
            pad_top = int(mouth_h * 0.6)
            pad_bottom = int(mouth_h * 0.9)

            min_x = max(0, min_x - pad_x)
            min_y = max(0, min_y - pad_top)
            max_x = min(frame.shape[1], max_x + pad_x)
            max_y = min(frame.shape[0], max_y + pad_bottom)
            
            # Extract the mouth area from the frame using the calculated bounding box
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

        return mouth_cutout, (min_x, min_y, max_x, max_y)



    # Eye restoration: mirror of the mouth restore, but for the two eyes as
    # SEPARATE tight boxes (Opsi A). Instead of hard-coding 2d106 eye indices
    # (which risks pasting in the wrong place if the index convention differs),
    # the eye landmark points are found geometrically: of the 106 points, take
    # those closest to each detector kps eye centre. This is robust to the exact
    # index layout and still excludes brows (brow points sit well above the eye
    # centre, outside the selection radius). Returns two (cutout, box) pairs.
    def create_eyes_masks(self, face: Face, frame: Frame):
        results = []
        landmarks = getattr(face, 'landmark_2d_106', None)
        kps = getattr(face, 'kps', None)
        if landmarks is None or kps is None:
            return results
        kps = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
        if kps.shape[0] < 2:
            return results
        lm = np.asarray(landmarks, dtype=np.float32)
        left_eye_c, right_eye_c = kps[0], kps[1]
        iod = float(np.linalg.norm(left_eye_c - right_eye_c)) + 1e-6
        # selection radius: tight enough to stay on the eye, not the brow/lid edge
        radius = 0.22 * iod
        for eye_c in (left_eye_c, right_eye_c):
            d = np.linalg.norm(lm - eye_c, axis=1)
            sel = lm[d < radius]
            if sel.shape[0] < 3:
                # fallback: nearest 6 points to the eye centre
                order = np.argsort(d)[:6]
                sel = lm[order]
            min_x, min_y = np.min(sel, axis=0)
            max_x, max_y = np.max(sel, axis=0)
            eye_w = max(1, int(max_x - min_x))
            eye_h = max(1, int(max_y - min_y))
            # Looser padding than before (0.25 -> 0.45). FaceFusion feathers the
            # whole eye-region mask heavily (mask_blur ~0.3-0.5) rather than using
            # a hard box; a soft edge needs ROOM to fade, so the box is widened a
            # bit and the actual restored ellipse stays inside it. Brows are still
            # excluded: even at 0.45 the box top stays below the brow because the
            # eye landmarks themselves sit well under it.
            pad_x = int(eye_w * 0.45)
            pad_y = int(eye_h * 0.45)
            min_x = max(0, int(min_x) - pad_x)
            min_y = max(0, int(min_y) - pad_y)
            max_x = min(frame.shape[1], int(max_x) + pad_x)
            max_y = min(frame.shape[0], int(max_y) + pad_y)
            if max_x - min_x < 2 or max_y - min_y < 2:
                continue
            cutout = frame[min_y:max_y, min_x:max_x].copy()
            results.append((cutout, (min_x, min_y, max_x, max_y)))
        return results

    def create_soft_eye_mask(self, shape):
        """Heavily-feathered elliptical mask for eye restoration.

        Unlike create_feathered_mask (feather clamped to <=30px and tuned for the
        large mouth box), this fills a smaller inner ellipse and applies a Gaussian
        blur whose radius scales with the box size (~35% of the smaller side). The
        result is a wide, gradual edge so the boundary stays invisible even when
        the face/ER is jittering -- the FaceFusion approach of leaning on a soft
        mask edge rather than a crisp box."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        # inner ellipse noticeably smaller than the box so the blur has room to
        # fade to zero well before the box edge (no hard cutoff at the border)
        ax_w = max(1, int(w * 0.26))
        ax_h = max(1, int(h * 0.26))
        cv2.ellipse(mask, (w // 2, h // 2), (ax_w, ax_h), 0, 0, 360, 1.0, -1)
        # blur radius ~50% of the smaller side, odd kernel, generous minimum
        k = max(9, int(min(w, h) * 0.5))
        if k % 2 == 0:
            k += 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        m = float(mask.max())
        if m > 1e-6:
            mask /= m
        return mask

    def apply_eyes_area(self, frame: np.ndarray, eyes):
        # Same color-transfer + blend as the mouth path, but with the dedicated
        # soft eye mask so the edge is wide and invisible under motion.
        for cutout, box in eyes:
            min_x, min_y, max_x, max_y = box
            bw, bh = max_x - min_x, max_y - min_y
            if cutout is None or bw < 2 or bh < 2:
                continue
            try:
                resized = cv2.resize(cutout, (bw, bh))
                roi = frame[min_y:max_y, min_x:max_x]
                if roi.shape != resized.shape:
                    resized = cv2.resize(resized, (roi.shape[1], roi.shape[0]))
                corrected = self.apply_color_transfer(resized, roi)
                mask = self.create_soft_eye_mask(resized.shape)[:, :, np.newaxis]
                blended = (corrected * mask + roi * (1 - mask)).astype(np.uint8)
                frame[min_y:max_y, min_x:max_x] = blended
            except Exception as e:
                print(f'Error {e}')
        return frame

    def create_feathered_mask(self, shape, feather_amount=30):
        mask = np.zeros(shape[:2], dtype=np.float32)
        center = (shape[1] // 2, shape[0] // 2)
        cv2.ellipse(mask, center, (shape[1] // 2 - feather_amount, shape[0] // 2 - feather_amount), 
                    0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 0)
        return mask / np.max(mask)

    def apply_mouth_area(self, frame: np.ndarray, mouth_cutout: np.ndarray, mouth_box: tuple) -> np.ndarray:
        min_x, min_y, max_x, max_y = mouth_box
        box_width = max_x - min_x
        box_height = max_y - min_y
        

        # Resize the mouth cutout to match the mouth box size
        if mouth_cutout is None or box_width is None or box_height is None:
            return frame
        try:
            resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
            
            # Extract the region of interest (ROI) from the target frame
            roi = frame[min_y:max_y, min_x:max_x]
            
            # Ensure the ROI and resized_mouth_cutout have the same shape
            if roi.shape != resized_mouth_cutout.shape:
                resized_mouth_cutout = cv2.resize(resized_mouth_cutout, (roi.shape[1], roi.shape[0]))
            
            # Apply color transfer from ROI to mouth cutout
            color_corrected_mouth = self.apply_color_transfer(resized_mouth_cutout, roi)
            
            # Create a feathered mask with increased feather amount
            feather_amount = min(30, box_width // 15, box_height // 15)
            mask = self.create_feathered_mask(resized_mouth_cutout.shape, feather_amount)
            
            # Blend the color-corrected mouth cutout with the ROI using the feathered mask
            mask = mask[:,:,np.newaxis]  # Add channel dimension to mask
            blended = (color_corrected_mouth * mask + roi * (1 - mask)).astype(np.uint8)
            
            # Place the blended result back into the frame
            frame[min_y:max_y, min_x:max_x] = blended
        except Exception as e:
            print(f'Error {e}')
            pass

        return frame

    def apply_color_transfer(self, source, target):
        """
        Apply color transfer from target to source image
        """
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        source_mean, source_std = cv2.meanStdDev(source)
        target_mean, target_std = cv2.meanStdDev(target)

        # Reshape mean and std to be broadcastable
        source_mean = source_mean.reshape(1, 1, 3)
        source_std = source_std.reshape(1, 1, 3)
        target_mean = target_mean.reshape(1, 1, 3)
        target_std = target_std.reshape(1, 1, 3)

        # Perform the color transfer. Guard against a (near-)zero source std --
        # a uniform patch (can happen on a small dark eye cutout) would divide by
        # zero and produce NaNs, which then get silently caught upstream and drop
        # the restore. Clamp the source std to a small floor so the scale stays
        # finite (colour just isn't stretched when there's nothing to stretch).
        source_std = np.maximum(source_std, 1e-3)
        source = (source - source_mean) * (target_std / source_std) + target_mean
        return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)



    def release_resources(self, keep_models=False):
        if keep_models:
            # park the processors (and their loaded sessions) for the next run
            with _idle_lock:
                for p in self.processors:
                    previous = _idle_processors.get(_plugin_key(p))
                    if previous is not None and previous is not p:
                        previous.Release()
                    _idle_processors[_plugin_key(p)] = p
        else:
            for p in self.processors:
                p.Release()
            _release_idle_processors()
        self.processors.clear()
        if self.videowriter is not None:
            self.videowriter.close()
        if self.streamwriter is not None:
            self.streamwriter.Close()

