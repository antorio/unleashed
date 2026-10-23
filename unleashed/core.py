#!/usr/bin/env python3

import os
import sys
import shutil
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from typing import List
import platform
import signal
try:
    import torch        # CUDA device selection only; a CPU-only install (Intel Mac) has none
except ImportError:
    torch = None
import onnxruntime
import pathlib
import threading
import argparse

from time import time

import unleashed.globals
import unleashed.metadata
import unleashed.utilities as util
import unleashed.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from unleashed.face_util import extract_face_images
from unleashed.ProcessEntry import ProcessEntry
from unleashed.ProcessMgr import ProcessMgr
from unleashed import onnx_guard
from unleashed.ProcessOptions import ProcessOptions
from unleashed.capturer import get_video_frame_total

clip_text = None

call_display_ui = None

process_mgr = None


if 'ROCMExecutionProvider' in unleashed.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    unleashed.globals.headless = False

    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('--server_share', help='Public server', dest='server_share', action='store_true', default=False)
    program.add_argument('--cuda_device_id', help='Index of the cuda gpu to use', dest='cuda_device_id', type=int, default=0)
    unleashed.globals.startup_args = program.parse_args()
    # Always enable all processors when using GUI
    unleashed.globals.frame_processors = ['face_swapper', 'face_enhancer']


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    list_providers = [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]
    
    try:
        for i in range(len(list_providers)):
            if list_providers[i] == 'CUDAExecutionProvider':
                list_providers[i] = ('CUDAExecutionProvider', {'device_id': unleashed.globals.cuda_device_id})
                if torch is not None:
                    torch.cuda.set_device(unleashed.globals.cuda_device_id)
                break
    except:
        pass

    return list_providers
    


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in unleashed.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in unleashed.globals.execution_providers:
        return 1
    return 8


_original_data_limit = None


def limit_resources() -> None:
    """Settings > Max. Memory (GB, 0 = no limit), applied at every Start.

    macOS does not enforce RLIMIT_DATA for the allocations that matter here
    (the old code also computed GB x 1024**6, so 8 or more crashed every
    render with OverflowError and 1-7 did nothing): skipped there. Elsewhere
    only the soft limit is set, so a larger value or 0 later can undo it
    without root, and 0 puts back what the process started with."""
    global _original_data_limit
    system = platform.system().lower()
    gb = int(unleashed.globals.max_memory or 0)
    try:
        if system == 'darwin':
            return
        if system == 'windows':
            if gb:
                import ctypes
                memory = gb * 1024 ** 3
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
            return
        import resource
        if _original_data_limit is None:
            _original_data_limit = resource.getrlimit(resource.RLIMIT_DATA)
        soft, hard = _original_data_limit
        if gb:
            memory = gb * 1024 ** 3
            if hard != resource.RLIM_INFINITY:
                memory = min(memory, hard)
            resource.setrlimit(resource.RLIMIT_DATA, (memory, hard))
        else:
            resource.setrlimit(resource.RLIMIT_DATA, (soft, hard))
    except (ValueError, OverflowError, OSError) as e:
        print(f'[memory] limit of {gb} GB not applied: {e}')



def release_resources(keep_models=False) -> None:
    import gc
    global process_mgr

    if process_mgr is not None:
        # keep_models: the loaded model sessions are parked and picked up by the
        # next preview/render instead of being rebuilt (see ProcessMgr)
        process_mgr.release_resources(keep_models=keep_models)
        process_mgr = None

    gc.collect()
    # if 'CUDAExecutionProvider' in unleashed.globals.execution_providers and torch.cuda.is_available():
    #     with torch.cuda.device('cuda'):
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False

    # All model weights are hosted in ONE HuggingFace repo (mirrored there from
    # the original countfloyd/deepfake + facefusion-assets sources). HF's CDN is
    # reliable from Colab/datacenter IPs -- unlike GitHub release assets, which
    # rate-limit those IPs and were the cause of the "download stuck from Colab
    # but works in a browser" hangs. hf_hub_download also resumes + retries, so a
    # dropped connection no longer restarts from zero.
    HF_REPO = 'antorio/runleashed-models'
    models_root = util.resolve_relative_path('../models')

    # (filename in the HF repo root, local subdir under ../models)
    model_files = [
        ('inswapper_128.onnx', ''),
        ('GFPGANv1.4.onnx', ''),
        ('restoreformer_plus_plus.onnx', ''),
        ('xseg.onnx', ''),
        ('xseg_1.onnx', ''),
        ('xseg_2.onnx', ''),
        ('xseg_3.onnx', ''),
        ('bisenet_resnet_34.onnx', ''),
        ('2dfan4.onnx', ''),
        ('live_portrait_feature_extractor.onnx', 'liveportrait'),
        ('live_portrait_motion_extractor.onnx', 'liveportrait'),
        ('live_portrait_generator.onnx', 'liveportrait'),
        ('live_portrait_stitcher.onnx', 'liveportrait'),
        ('rd64-uni-refined.pth', 'CLIP'),
        ('CodeFormerv0.1.onnx', 'CodeFormer'),
    ]

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        hf_hub_download = None

    for filename, subdir in model_files:
        target_dir = os.path.join(models_root, subdir) if subdir else models_root
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(target_path):
            continue
        got = False
        if hf_hub_download is not None:
            try:
                # straight into ../models[/subdir]/<filename>, the path the
                # loaders expect (no second copy in the HF cache: ~2.5 GB)
                got_path = hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=target_dir)
                if os.path.abspath(got_path) != os.path.abspath(target_path):
                    shutil.copyfile(got_path, target_path)
                got = True
            except Exception as e:
                update_status(f'HF download failed for {filename} ({e}); falling back to direct URL')
        if not got:
            # Fallback: direct resolve URL from the same HF repo (still HF CDN).
            util.conditional_download(target_dir, [f'https://huggingface.co/{HF_REPO}/resolve/main/{filename}'])

    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True

def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)




def start() -> None:
    if unleashed.globals.headless:
        print('Headless mode currently unsupported - starting UI!')
        # faces = extract_face_images(unleashed.globals.source_path,  (False, 0))
        # unleashed.globals.INPUT_FACES.append(faces[unleashed.globals.source_face_index])
        # faces = extract_face_images(unleashed.globals.target_path,  (False, util.has_image_extension(unleashed.globals.target_path)))
        # unleashed.globals.TARGET_FACES.append(faces[unleashed.globals.target_face_index])
        # if 'face_enhancer' in unleashed.globals.frame_processors:
        #     unleashed.globals.selected_enhancer = 'GFPGAN'
       
    batch_process_regular(None, False, None)


def get_processing_plugins(masking_engine):
    processors = {  "faceswap": {}}

    # Expression restorer runs right after the swap.
    if getattr(unleashed.globals, 'expression_restorer', False):
        processors.update({"expression_restorer": {}})

    # Occlusion mask runs AFTER the expression restorer (so the ER never warps the
    # restored occluders like hands/hair). The mask/enhancer order is controlled
    # SOLELY by the user's 'mask_after_enhancer' toggle, for EVERY enhancer
    # (GFPGAN, CodeFormer, RestoreFormer++): off = mask before the enhancer;
    # on (the default in globals.py) = mask runs last.
    # Note: RestoreFormer++ / CodeFormer are codebook restorers that can colour-burn
    # when they enhance a crop that still contains a restored occluder (non-face
    # pixels). If you use them on footage WITH occluders and see colour burn, tick
    # 'mask after enhancer' so they only ever see the clean swapped face. We do NOT
    # force this anymore -- e.g. frontal / occluder-free clips don't need it, and
    # forcing it added slight jitter -- so the choice is yours.
    mask_after_enhancer = getattr(unleashed.globals, 'mask_after_enhancer', False)
    if masking_engine is not None and not mask_after_enhancer:
        processors.update({masking_engine: {}})

    if unleashed.globals.selected_enhancer == 'GFPGAN':
        processors.update({"gfpgan": {}})
    elif unleashed.globals.selected_enhancer == 'Codeformer':
        processors.update({"codeformer": {}})
    elif unleashed.globals.selected_enhancer == 'Restoreformer++':
        processors.update({"restoreformer++": {}})

    if masking_engine is not None and mask_after_enhancer:
        processors.update({masking_engine: {}})
    return processors


# One ProcessMgr serves the preview and the render. The lock keeps a preview
# from being re-initialised halfway through a frame, and a render from starting
# under a preview; render_active keeps any preview (and the settings it would
# apply) away from a running render, whose per-frame globals and ProcessMgr
# state (frame sequencer, stabiliser, tracks) must not change.
_swap_lock = threading.RLock()
render_active = False
mask_view_mgr = None
# Stop pressed before batch_process starts (the render still waits for a
# preview or loads its models): batch_process sets processing = True, which
# used to wipe that Stop out. Set by the UI's Stop, cleared at its Start.
stop_requested = False


def preview_locked(work):
    """Run work() for a preview unless a render is running (or starting).
    Returns (ran, result); never waits for a render."""
    if not _swap_lock.acquire(blocking=False):
        return False, None
    try:
        if render_active:
            return False, None
        return True, work()
    finally:
        _swap_lock.release()


def live_swap(frame, options):
    global process_mgr

    if frame is None:
        return frame

    with _swap_lock:
        if process_mgr is None:
            process_mgr = ProcessMgr(None)
        process_mgr.initialize(unleashed.globals.INPUT_FACESETS, unleashed.globals.TARGET_FACES, options)
        newframe = process_mgr.process_frame(frame)
    if newframe is None:
        return frame
    return newframe


def mask_view(frame, options):
    """The frame with the area the swap replaces tinted green and occluders
    (kept original) cut out of it, no swap. Its own ProcessMgr, so the main
    one keeps its restorer / enhancer loaded."""
    global mask_view_mgr
    if frame is None:
        return frame
    with _swap_lock:
        if mask_view_mgr is None:
            mask_view_mgr = ProcessMgr(None)
        mask_view_mgr.initialize(unleashed.globals.INPUT_FACESETS, unleashed.globals.TARGET_FACES, options, release_idle=False)
        newframe = mask_view_mgr.process_frame(frame)
    return frame if newframe is None else newframe


def batch_process_regular(swap_model, output_method, files:list[ProcessEntry], masking_engine:str, new_clip_text:str, use_new_method, imagemask, restore_original_mouth, restore_original_eyes, num_swap_steps, progress, selected_index = 0) -> None:
    global clip_text, process_mgr, render_active

    with _swap_lock:            # waits for a preview that is still running
        render_active = True
    try:
        _batch_process_regular(swap_model, output_method, files, masking_engine, new_clip_text, use_new_method, imagemask,
                               restore_original_mouth, restore_original_eyes, num_swap_steps, progress, selected_index)
    finally:
        render_active = False


def _batch_process_regular(swap_model, output_method, files, masking_engine, new_clip_text, use_new_method, imagemask, restore_original_mouth, restore_original_eyes, num_swap_steps, progress, selected_index):
    global clip_text, process_mgr

    with _swap_lock:
        release_resources(keep_models=True)
        limit_resources()
        if process_mgr is None:
            process_mgr = ProcessMgr(progress)
        mask = imagemask["layers"][0] if isinstance(imagemask, dict) else imagemask
        if len(unleashed.globals.INPUT_FACESETS) <= selected_index:
            selected_index = 0
        options = ProcessOptions(swap_model, get_processing_plugins(masking_engine), unleashed.globals.distance_threshold, unleashed.globals.blend_ratio,
                                  unleashed.globals.face_swap_mode, selected_index, new_clip_text, mask, num_swap_steps,
                                  unleashed.globals.subsample_size, False, restore_original_mouth, restore_original_eyes=restore_original_eyes)
        process_mgr.initialize(unleashed.globals.INPUT_FACESETS, unleashed.globals.TARGET_FACES, options)
    batch_process(output_method, files, use_new_method)
    return

def batch_process(output_method, files:list[ProcessEntry], use_new_method) -> None:
    global clip_text, process_mgr

    unleashed.globals.processing = True
    if stop_requested:
        unleashed.globals.processing = False
        end_processing('Processing stopped!')
        return
    run_started = time()

    def fresh(path):
        # written by this run (a file of an earlier run with the same name,
        # e.g. an output template without {time}, does not count)
        return bool(path) and os.path.isfile(path) and os.path.getmtime(path) >= run_started - 1

    def ok(path):
        # a real file: ffmpeg creates the output before it fails, leaving 0 bytes
        return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0

    def playable(path):
        # its first frame can be read (a broken intermediate must not be saved
        # as the result, e.g. without sound)
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            return cap.isOpened() and cap.read()[0]
        finally:
            cap.release()

    # limit threads for some providers
    max_threads = suggest_execution_threads()
    if max_threads == 1:
        unleashed.globals.execution_threads = 1

    imagefiles:list[ProcessEntry] = []
    videofiles:list[ProcessEntry] = []
           
    update_status('Sorting videos/images')

    def _key(path):
        # one file, however it is spelled: case-insensitive disks (macOS,
        # Windows) take PHOTO.PNG and photo.png for the same file
        return os.path.normcase(os.path.realpath(path)).lower()

    inputs = {_key(f.filename) for f in files}
    used = set()

    def taken(path):
        return _key(path) in used or _key(path) in inputs

    def unique(path, marker=''):
        # two targets with the same file name (other folders, or the same second
        # in {time}) would write one output, and a name may be one of this run's
        # inputs: only then add _2, _3, ...
        if not taken(path):
            used.add(_key(path))
            return path
        stem, ext = os.path.splitext(path)
        if marker and stem.endswith(marker):
            stem = stem[:-len(marker)]
        n = 2
        while taken(f'{stem}_{n}{marker}{ext}'):
            n += 1
        path = f'{stem}_{n}{marker}{ext}'
        used.add(_key(path))
        return path

    def final_name(fullname, ext, index):
        """The output file of one target, decided BEFORE rendering: through the
        output template, never one of this run's inputs (a template '{file}'
        with the input in the output folder wrote over the original) and never
        another output of this run (a template without {file}/{i} made every
        video of a run the same file)."""
        name = util.replace_template(util.get_destfilename_from_path(fullname, unleashed.globals.output_path, ext), index=index)
        if _key(name) in inputs:
            stem, e = os.path.splitext(name)
            name = f'{stem}_swapped{e}'
        return unique(name)

    video_index = 0                     # {i} counts the videos among themselves, as before
    for index, f in enumerate(files):
        fullname = f.filename
        if util.has_image_extension(fullname):
            destination = final_name(fullname, f'.{unleashed.globals.CFG.output_image_format}', index)
            pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
            f.finalname = destination
            imagefiles.append(f)

        elif util.is_video(fullname) or util.has_extension(fullname, ['gif']):
            ext = '.gif' if util.has_extension(fullname, ['gif']) else f'.{unleashed.globals.CFG.output_video_format}'
            f.destination = final_name(fullname, ext, video_index)
            video_index += 1
            pathlib.Path(os.path.dirname(f.destination)).mkdir(parents=True, exist_ok=True)
            # the render goes to an intermediate <name>__temp file first; named
            # after the target only (not the template), so the next run of the
            # same file reuses it instead of leaving one more behind
            f.finalname = unique(util.get_destfilename_from_path(fullname, unleashed.globals.output_path,
                                                                  f'__temp.{unleashed.globals.CFG.output_video_format}'), '__temp')
            videofiles.append(f)



    if(len(imagefiles) > 0):
        update_status('Processing image(s)')
        gpu_before = onnx_guard.gpu_event_count()
        # a painted keep-original mask belongs to the file it was painted on
        plain = [f for f in imagefiles if getattr(f, 'imagemask', None) is None]
        masked = [f for f in imagefiles if getattr(f, 'imagemask', None) is not None]
        if plain:
            process_mgr.set_manual_mask(None)
            process_mgr.run_batch([f.filename for f in plain], [f.finalname for f in plain], unleashed.globals.execution_threads)
        for f in masked:
            if not unleashed.globals.processing:
                break
            process_mgr.set_manual_mask(f.imagemask)
            process_mgr.run_batch([f.filename], [f.finalname], unleashed.globals.execution_threads)
        process_mgr.set_manual_mask(None)
        events = onnx_guard.gpu_events_since(gpu_before)
        for f in imagefiles:
            # "Drop the frame" writes no file for an image
            f.completed = fresh(f.finalname)
            name = os.path.basename(f.filename)
            note_gpu_errors(f, [e for e in events if e['where'] == name], name)

    if(len(videofiles) > 0):
        for index,v in enumerate(videofiles):
            if not unleashed.globals.processing:
                end_processing('Processing stopped!')
                return
            fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
            if v.endframe == 0:
                v.endframe = get_video_frame_total(v.filename)

            is_streaming_only = output_method == "Virtual Camera"
            if is_streaming_only == False:
                update_status(f'Creating {os.path.basename(v.finalname)} with {fps} FPS...')

            start_processing = time()
            gpu_before = onnx_guard.gpu_event_count()
            if hasattr(v, 'imagemask'):
                process_mgr.set_manual_mask(v.imagemask)
            if is_streaming_only == False and unleashed.globals.keep_frames or not use_new_method:
                # this video's frame folder starts empty: frames kept from an
                # earlier or stopped run were swapped again and encoded too
                shutil.rmtree(util.get_temp_directory_path(v.filename), ignore_errors=True)
                util.create_temp(v.filename)
                update_status('Extracting frames...')
                ffmpeg.extract_frames(v.filename,v.startframe,v.endframe, fps)
                if not unleashed.globals.processing:
                    util.clean_temp(v.filename)             # the frames, unless Keep frames
                    end_processing('Processing stopped!')
                    return

                temp_frame_paths = util.get_temp_frame_paths(v.filename)
                if not temp_frame_paths:
                    v.failure = 'no frames could be read from it'
                    update_status(f'Failed processing {os.path.basename(v.filename)}: no frames')
                    util.clean_temp(v.filename)
                    continue
                process_mgr.run_batch(temp_frame_paths, temp_frame_paths, unleashed.globals.execution_threads)
                if not unleashed.globals.processing:
                    util.clean_temp(v.filename)
                    end_processing('Processing stopped!')
                    return
                if unleashed.globals.wait_after_extraction:
                    extract_path = os.path.dirname(temp_frame_paths[0])
                    util.open_folder(extract_path)
                    input("Press any key to continue...")
                    print("Resorting frames to create video")
                    util.sort_rename_frames(extract_path)                                    
                
                if ffmpeg.create_video(v.filename, v.finalname, fps) and ok(v.finalname):
                    if not unleashed.globals.keep_frames:
                        util.delete_temp_frames(temp_frame_paths[0])
                else:
                    # keep the swapped frames: they are the only result
                    v.failure = f'the video could not be made (swapped frames kept in {os.path.dirname(temp_frame_paths[0])})'
                    if os.path.isfile(v.finalname):
                        os.remove(v.finalname)
            else:
                if util.has_extension(v.filename, ['gif']):
                    skip_audio = True
                else:
                    skip_audio = unleashed.globals.skip_audio
                process_mgr.run_batch_inmem(output_method, v.filename, v.finalname, v.startframe, v.endframe, fps,unleashed.globals.execution_threads)
                
            if not unleashed.globals.processing:
                end_processing('Processing stopped!')
                return
            
            video_file_name = v.finalname
            destination = v.destination
            if ok(video_file_name) and not playable(video_file_name):
                v.failure = f'the rendered video is damaged ({os.path.basename(video_file_name)} kept)'
            elif ok(video_file_name):
                if util.has_extension(v.filename, ['gif']):
                    update_status('Creating final GIF')
                    if ffmpeg.create_gif_from_video(video_file_name, destination) and ok(destination):
                        os.remove(video_file_name)
                    else:
                        v.failure = f'the GIF could not be made (the video is kept as {os.path.basename(video_file_name)})'
                elif unleashed.globals.skip_audio:
                    shutil.move(video_file_name, destination)
                else:
                    # the rendered video is only deleted once the file with
                    # sound is really there: ffmpeg leaves a 0-byte file when
                    # the source's sound does not fit the container
                    if not (ffmpeg.restore_audio(video_file_name, v.filename, v.startframe, v.endframe, destination)
                            and ok(destination)):
                        # a codec the container can hold (WebM takes no AAC)
                        codec = 'libopus' if destination.lower().endswith('.webm') else 'aac'
                        update_status(f'Copying the sound failed, converting it ({codec})')
                        if not (ffmpeg.restore_audio(video_file_name, v.filename, v.startframe, v.endframe, destination,
                                                     audio_codec=codec) and ok(destination)):
                            if os.path.isfile(destination):
                                os.remove(destination)
                            shutil.move(video_file_name, destination)
                            v.no_audio = True
                    if os.path.isfile(video_file_name):
                        os.remove(video_file_name)
                # the finished file (finalname was the intermediate __temp file,
                # which is gone now): this is what the UI shows as the result
                if ok(destination) and fresh(destination):
                    v.finalname = destination
                    v.completed = True

            elif is_streaming_only == False:
                if not getattr(v, 'failure', ''):
                    v.failure = 'nothing was written'
                update_status(f'Failed processing {os.path.basename(v.finalname)}!')
            note_gpu_errors(v, onnx_guard.gpu_events_since(gpu_before), os.path.basename(v.filename))
            elapsed_time = time() - start_processing
            average_fps = (v.endframe - v.startframe) / elapsed_time
            update_status(f'\nProcessing {os.path.basename(destination)} took {elapsed_time:.2f} secs, {average_fps:.2f} frames/s')
    end_processing('Finished')


def note_gpu_errors(entry, events, name):
    """The GPU errors (onnx_guard) while one file was processed: kept on the
    entry for the Face Swap tab's report, summed up in the console."""
    entry.gpu_recovered = sum(1 for e in events if e['recovered'])
    # one place once (a frame's analysis runs again for the rotated tries)
    entry.gpu_failed = list(dict.fromkeys(e['where'] or e['model'] for e in events if not e['recovered']))
    if entry.gpu_failed:
        print(f"[gpu] {name}: {len(entry.gpu_failed)} GPU error(s) NOT recovered by a retry, at "
              f"{', '.join(entry.gpu_failed[:20])}{' ...' if len(entry.gpu_failed) > 20 else ''}: "
              f"check those frames (faces there may be left unswapped)", flush=True)
    elif entry.gpu_recovered:
        print(f"[gpu] {name}: {entry.gpu_recovered} GPU error(s), all recovered by a retry", flush=True)


def end_processing(msg:str):
    update_status(msg)
    unleashed.globals.target_folder_path = None
    release_resources(keep_models=True)


def destroy() -> None:
    if unleashed.globals.target_path:
        util.clean_temp(unleashed.globals.target_path)
    release_resources()        
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    unleashed.globals.CFG = Settings('config.yaml')
    unleashed.globals.cuda_device_id = unleashed.globals.startup_args.cuda_device_id
    unleashed.globals.execution_threads = unleashed.globals.CFG.max_threads
    unleashed.globals.video_encoder = unleashed.globals.CFG.output_video_codec
    unleashed.globals.video_quality = unleashed.globals.CFG.video_quality
    unleashed.globals.max_memory = unleashed.globals.CFG.memory_limit if unleashed.globals.CFG.memory_limit > 0 else None
    if unleashed.globals.startup_args.server_share:
        unleashed.globals.CFG.server_share = True       # this start only: server_share_saved keeps the config's
    main.run()
