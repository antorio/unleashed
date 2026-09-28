"""Face Swap tab: state and logic, no layout (ui/tabs/faceswap_tab.py builds
the page and wires it to these functions).

State lives on the server, module-level (one user at a time, like the rest of
the app): the sources (unleashed.globals.INPUT_FACESETS + thumbnails + captions +
which one is active), the target files (with their frame range, output frame
rate and painted mask), the people to replace (unleashed.globals.TARGET_FACES), and
the settings last applied from the panel.

Settings are applied from the panel at every preview refresh and once at Start
(apply_settings), inside the engine's lock and never while a render runs, so a
render keeps the settings it started with. The panel is refilled from `values`
on page load, so a reload shows the real state.
"""
import json
import os
import random
import shutil
import tempfile
import threading

import cv2
import numpy as np

import unleashed.globals
import unleashed.utilities as util
import ui.globals
from unleashed.FaceSet import FaceSet
from unleashed.ProcessEntry import ProcessEntry
from unleashed.ProcessOptions import ProcessOptions

G = unleashed.globals
SWAP_MODEL = "InSwapper 128"        # the only model; ProcessOptions reads the output size (128) from it

# ----------------------------------------------------------------------------- choices

MODES = {                           # label -> ProcessMgr swap mode
    'Largest face': 'first',
    'All faces': 'all',
    'Specific people': 'selected',
    'All women': 'all_female',
    'All men': 'all_male',
    'One source per face (left to right)': 'all_input',
}
NO_FACE = {                         # label -> unleashed.globals.no_face_action
    'Keep the original frame': 0,
    'Try rotated, else keep the original': 1,
    'Drop the frame': 2,
}
MASK_ENGINES = {
    'None': None,
    'DFL XSeg': 'mask_xseg',
    'FF XSeg 1': 'mask_ffxseg1',
    'FF XSeg 2': 'mask_ffxseg2',
    'FF XSeg 3': 'mask_ffxseg3',
    'FF XSeg (combined)': 'mask_ffxseg_many',
    'Face Parser (BiSeNet)': 'mask_faceparser',
    'Clip2Seg (by text)': 'mask_clip2seg',
}
ENHANCERS = {'None': 'None', 'GFPGAN': 'GFPGAN', 'CodeFormer': 'Codeformer', 'RestoreFormer++': 'Restoreformer++'}
RESOLUTIONS = ['128px', '256px', '512px', '768px', '1024px']
METHOD_MEMORY, METHOD_EXTRACT = 'In memory (recommended)', 'Extract frames to disk'
MULTI_ANGLE = {'Off': 'off', 'Only when no upright face is found': 'fallback', 'Always (slower)': 'always'}
VIEWS = ['Original', 'Swapped', 'Side by side', 'Mask']

# accepted exactly as the render accepts them (unleashed.core.batch_process)
def is_target_file(path):
    return util.has_image_extension(path) or util.is_video(path) or util.has_extension(path, ['gif'])


def is_source_file(path):
    return path.lower().endswith('.fsz') or util.has_image_extension(path)


# ----------------------------------------------------------------------------- settings

# key -> initial value: the old panel's defaults (the engine globals for the
# settings that were bound to one), so the default output is unchanged --
# except what the user changed on 26 Sep (unleashed/globals.py): Match tolerance
# 0.65 -> 0.8, ER on with strength 100 (was off / 80), Extend swap to chin
# and Face-aligned edges on
FACTORY = {
    # faces to replace
    'mode': 'Largest face', 'tolerance': G.distance_threshold,
    # swap
    'resolution': '256px', 'passes': 1, 'passes_keep': G.passes_keep_look,
    'identity': G.identity_strength, 'face_shape': G.face_shape_strength,
    # expression
    'er': G.expression_restorer, 'er_strength': G.expression_restorer_factor,
    'er_eyes': G.expression_restore_eyes, 'er_mouth': G.expression_restore_mouth, 'er_brows': G.expression_restore_brows,
    'er_smoothing': G.expression_smoothing_strength, 'er_pose_lock': G.expression_pose_lock,
    'er_pose_gate': G.expression_pose_gate, 'er_structure': G.expression_keep_structure,
    'keep_mouth': False, 'keep_eyes': False,
    # occlusion, edges & blending
    'mask_engine': 'DFL XSeg', 'mask_objects': 'cup,hands,hair,banana',
    'erosion': G.mask_erosion_iterations, 'blur': G.mask_blur_size, 'to_chin': G.mask_bottom_to_chin,
    'mask_grow': G.occlusion_mask_grow, 'mask_soften': G.occlusion_mask_soften,
    'aligned_edges': G.mask_face_aligned, 'color_transfer': G.use_color_transfer,
    'crop_top': 0.0, 'crop_bottom': 0.0, 'crop_left': 0.0, 'crop_right': 0.0,
    # enhance
    'enhancer': 'None', 'enhancer_blend': 0.65, 'mask_after_enhancer': G.mask_after_enhancer,
    # detection & tracking
    'det_thresh': G.det_thresh, 'det_size': G.det_size,
    'multi_angle': next(k for k, v in MULTI_ANGLE.items() if v == G.multi_angle_detection_mode),
    'upright': G.angle0_bonus, 'autorotate': True,
    'lmk_align': G.use_landmark_alignment, 'hi_lmk': G.use_hi_landmarker,
    'lmk_gate': G.landmark_sanity_gate, 'lmk_gate_thr': G.landmark_sanity_threshold,
    'smoothing': G.landmark_smoothing, 'smoothing_strength': G.landmark_smoothing_strength,
    'smoothing_deadzone': G.landmark_smoothing_deadzone,
    # video output
    'method': METHOD_MEMORY, 'keep_frames': False, 'no_face': 'Try rotated, else keep the original',
    'skip_audio': False,
}
values = dict(FACTORY)
SETTING_KEYS = list(FACTORY)
_SETTINGS_FILE = 'unleashed_faceswap_settings.json'
_LEGACY_SETTINGS_FILE = 'runleashed_faceswap_settings.json'     # the name before v1.2


def _settings_path():
    """The saved defaults in the output folder. A file saved before v1.2 under
    the old name is renamed once, so the user's defaults are not lost."""
    folder = G.output_path or '.'
    path = os.path.join(folder, _SETTINGS_FILE)
    legacy = os.path.join(folder, _LEGACY_SETTINGS_FILE)
    if not os.path.isfile(path) and os.path.isfile(legacy):
        try:
            os.replace(legacy, path)
        except OSError:
            return legacy
    return path


def load_saved_defaults():
    """The user's saved defaults (in the output folder, so they survive a new
    Colab session), if any. Unknown or invalid entries are ignored."""
    try:
        with open(_settings_path(), encoding='utf-8') as fh:
            saved = json.load(fh)
    except Exception:
        return False
    choices = {'mode': MODES, 'no_face': NO_FACE, 'mask_engine': MASK_ENGINES, 'enhancer': ENHANCERS,
               'resolution': RESOLUTIONS, 'method': (METHOD_MEMORY, METHOD_EXTRACT), 'multi_angle': MULTI_ANGLE,
               'det_size': (320, 640, 1024)}
    for k, v in saved.items():
        if k not in FACTORY:
            continue
        d = FACTORY[k]
        if isinstance(d, bool):
            ok = isinstance(v, bool)
        elif isinstance(d, (int, float)):
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        else:
            ok = isinstance(v, str)
        if ok and k in choices:
            ok = v in choices[k]
        if ok:
            values[k] = v
    return True


def save_defaults():
    with open(_settings_path(), 'w', encoding='utf-8') as fh:
        json.dump({k: values[k] for k in SETTING_KEYS}, fh, indent=1)
    return _settings_path()


def has_saved_defaults():
    return os.path.isfile(_settings_path())


def load_my_defaults():
    """The saved defaults (on top of the factory ones, for keys the file does
    not have). False when there is no saved file."""
    if not has_saved_defaults():
        return False
    values.update(FACTORY)
    return load_saved_defaults()


def factory_defaults():
    """The panel back to the factory values. The saved defaults file stays
    (it loads at the next start); Save replaces it."""
    values.update(FACTORY)


def apply_settings(settings=None):
    """Store the panel's values and write them to the engine: globals, and the
    crop offsets onto every source. Call inside the engine lock, never while a
    render runs (unleashed.core.preview_locked / at Start)."""
    if settings:
        # None = an emptied number box: keep the value in use
        values.update({k: v for k, v in settings.items() if k in FACTORY and v is not None})
    v = values
    G.face_swap_mode = MODES[v['mode']]
    G.distance_threshold = float(v['tolerance'])
    G.subsample_size = int(str(v['resolution']).lower().replace('px', '').strip())
    G.identity_strength = float(v['identity'])
    G.face_shape_strength = float(v['face_shape'])
    G.expression_restorer = bool(v['er'])
    G.expression_restorer_factor = float(v['er_strength'])
    G.expression_restore_eyes = bool(v['er_eyes'])
    G.expression_restore_mouth = bool(v['er_mouth'])
    G.expression_restore_brows = bool(v['er_brows'])
    G.expression_smoothing_strength = float(v['er_smoothing'])
    G.expression_pose_lock = bool(v['er_pose_lock'])
    G.expression_pose_gate = bool(v['er_pose_gate'])
    G.expression_keep_structure = bool(v['er_structure'])
    G.passes_keep_look = bool(v['passes_keep'])
    G.occlusion_mask_grow = float(v['mask_grow'])
    G.occlusion_mask_soften = float(v['mask_soften'])
    G.mask_erosion_iterations = int(v['erosion'])
    G.mask_blur_size = int(v['blur'])
    G.mask_bottom_to_chin = bool(v['to_chin'])
    G.mask_face_aligned = bool(v['aligned_edges'])
    G.use_color_transfer = bool(v['color_transfer'])
    G.selected_enhancer = ENHANCERS[v['enhancer']]
    G.blend_ratio = float(v['enhancer_blend'])
    G.mask_after_enhancer = bool(v['mask_after_enhancer'])
    G.det_thresh = float(v['det_thresh'])
    G.det_size = int(v['det_size'])
    G.multi_angle_detection_mode = MULTI_ANGLE[v['multi_angle']]
    G.angle0_bonus = float(v['upright'])
    G.autorotate_faces = bool(v['autorotate'])
    G.use_landmark_alignment = bool(v['lmk_align'])
    G.use_hi_landmarker = bool(v['hi_lmk'])
    G.landmark_sanity_gate = bool(v['lmk_gate'])
    G.landmark_sanity_threshold = float(v['lmk_gate_thr'])
    G.landmark_smoothing = bool(v['smoothing'])
    G.landmark_smoothing_strength = float(v['smoothing_strength'])
    G.landmark_smoothing_deadzone = float(v['smoothing_deadzone'])
    G.no_face_action = NO_FACE[v['no_face']]
    G.skip_audio = bool(v['skip_audio'])
    G.keep_frames = bool(v['keep_frames']) and v['method'] == METHOD_EXTRACT
    G.vr_mode = False
    G.wait_after_extraction = False
    G.execution_threads = int(round(float(G.CFG.max_threads)))
    # crop offsets: the same for every source (top + bottom and left + right < 1)
    def crop(key, limit):
        try:
            return min(max(float(v[key] or 0), 0.0), limit)
        except (TypeError, ValueError):
            return 0.0
    top = crop('crop_top', 0.99)
    bottom = crop('crop_bottom', 0.99 - top)
    left = crop('crop_left', 0.99)
    right = crop('crop_right', 0.99 - left)
    for face_set in G.INPUT_FACESETS:
        # Settings > Faceset average mode / outlier threshold changed since
        # this faceset was averaged: average it again (it only happened when
        # the faceset was added, so the setting did nothing for it)
        if not face_set.average_is_current():
            face_set.reaverage()
        for face in face_set.faces:
            mo = list(getattr(face, 'mask_offsets', None) or (0, 0, 0, 0, 1, 20))
            face.mask_offsets = (top, bottom, left, right, mo[4], mo[5])


def mask_plugin():
    """(plugin name or None, warning or None) for the chosen occlusion mask."""
    plugin = MASK_ENGINES.get(values['mask_engine'])
    if plugin == 'mask_clip2seg':
        if not (values['mask_objects'] or '').strip():
            return None, 'Clip2Seg: type the objects to keep (e.g. hands, hair). No mask is used now.'
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            return None, 'Clip2Seg needs PyTorch, which is not installed here. No mask is used now.'
    return plugin, None


def build_options(mask_view=False):
    """ProcessOptions for the preview from the applied settings, with the
    shown file's painted mask (the render sets it per file itself). mask_view:
    the Mask view, no swap, every face, tinted green where a swap would
    replace the frame (occlusion mask, paste edges, kept mouth / eyes and
    painted areas stay untinted)."""
    from unleashed.core import get_processing_plugins
    plugin, _ = mask_plugin()
    t = target()
    painted = t['mask'] if t is not None else None
    keep_mouth, keep_eyes = bool(values['keep_mouth']), bool(values['keep_eyes'])
    if mask_view:
        processors = {plugin: {}} if plugin else {}
        return ProcessOptions(SWAP_MODEL, processors, G.distance_threshold, G.blend_ratio, 'all', 0,
                              values['mask_objects'], painted, 0, G.subsample_size, False, keep_mouth,
                              True, restore_original_eyes=keep_eyes)
    return ProcessOptions(SWAP_MODEL, get_processing_plugins(plugin), G.distance_threshold, G.blend_ratio,
                          G.face_swap_mode, active_source_index(), values['mask_objects'], painted,
                          int(values['passes']), G.subsample_size, False, keep_mouth,
                          restore_original_eyes=keep_eyes)


def summary(section):
    """A few words for a section's header."""
    v = values
    if section == 'sources':
        n = len(G.INPUT_FACESETS)
        return '' if n == 0 else f'{n} · in this order' if MODES[v['mode']] == 'all_input' else str(n)
    if section == 'targets':
        images = sum(1 for t in targets if t['kind'] == 'image')
        videos = len(targets) - images
        return ' · '.join(x for x in (f"{videos} video{'s' if videos != 1 else ''}" if videos else '',
                                      f"{images} image{'s' if images != 1 else ''}" if images else '') if x)
    if section == 'faces':
        return v['mode'] + (f' · {len(G.TARGET_FACES)} picked' if MODES[v['mode']] == 'selected' else '')
    if section == 'swap':
        parts = [v['resolution']]
        if int(v['passes']) > 1:
            parts.append(f"{int(v['passes'])} passes" + (' (kept)' if v['passes_keep'] else ''))
        if float(v['identity']) > 0:
            parts.append(f"identity {float(v['identity']):.2f}")
        if float(v['face_shape']) > 0:
            parts.append(f"face shape {float(v['face_shape']):.2f}")
        return ' · '.join(parts)
    if section == 'expression':
        parts = [f"on · {float(v['er_strength']):.0f}" if v['er'] else 'off']
        if v['keep_mouth'] or v['keep_eyes']:
            parts.append('target ' + ' + '.join(k for k, on in (('mouth', v['keep_mouth']), ('eyes', v['keep_eyes'])) if on))
        return ' · '.join(parts)
    if section == 'occlusion':
        parts = [v['mask_engine']]
        if MASK_ENGINES.get(v['mask_engine']) and (float(v['mask_grow']) or float(v['mask_soften'])):
            parts.append(f"edge {float(v['mask_grow']):+g}/{float(v['mask_soften']):g}")
        if any(t.get('mask') is not None for t in targets):
            parts.append('painted')
        return ' · '.join(parts)
    if section == 'enhance':
        return 'none' if v['enhancer'] == 'None' else f"{v['enhancer']} · {float(v['enhancer_blend']):.2f}"
    if section == 'detection':
        return f"{float(v['det_thresh']):.2f} · {v['det_size']} · " + ('68-pt' if v['lmk_align'] else '5-pt')
    if section == 'video':
        return ('in memory' if v['method'] == METHOD_MEMORY else 'extract frames') + ' · ' + {
            0: 'keep original', 1: 'try rotated', 2: 'drop frame'}[NO_FACE[v['no_face']]]
    return ''


# ----------------------------------------------------------------------------- sources

source_captions = []                 # parallel to unleashed.globals.INPUT_FACESETS / ui.globals.ui_input_thumbs
active_source = 0


def active_source_index():
    n = len(G.INPUT_FACESETS)
    return active_source if 0 <= active_source < n else 0


def _sync_sources():
    """Keep the captions aligned when something else changed the sources
    (Settings > Clean temp folder clears them)."""
    global active_source
    n = len(G.INPUT_FACESETS)
    del ui.globals.ui_input_thumbs[n:]
    del source_captions[n:]
    while len(ui.globals.ui_input_thumbs) < n:          # added without a thumbnail
        ui.globals.ui_input_thumbs.append(np.full((112, 112, 3), 128, np.uint8))
    while len(source_captions) < n:
        source_captions.append(f'source {len(source_captions) + 1}')
    if active_source >= n:
        active_source = max(0, n - 1)


def sources_gallery():
    """The source thumbnails, whole and without captions (the numbers of
    "One source per face" are drawn by CSS in a corner)."""
    _sync_sources()
    return list(ui.globals.ui_input_thumbs)


def combine_label():
    """The Combine button's text: how many photos it would blend."""
    k = len(same_person_photos())
    return f'Combine {k} photos into one source' if k >= 2 else 'Combine photos into one source'


def same_person_photos():
    """Indexes of the photo sources of the person in use (the one in use
    included), or [] when the source in use is a faceset. Same person: the
    faceset check's "other person?" threshold (unleashed/faceset_check.py)."""
    _sync_sources()
    if not G.INPUT_FACESETS:
        return []
    a = active_source_index()
    if not getattr(G.INPUT_FACESETS[a], 'from_photo', False):
        return []
    ref = _unit_embedding(G.INPUT_FACESETS[a])
    return [i for i, s in enumerate(G.INPUT_FACESETS)
            if getattr(s, 'from_photo', False) and float(_unit_embedding(s) @ ref) >= 0.40]


def add_sources(paths, progress=None):
    """Load photos (every face becomes a source) and .fsz facesets (one source
    each). Returns a list of per-file messages."""
    global active_source
    from unleashed.face_util import extract_face_images
    from unleashed.capturer import get_image_frame
    _sync_sources()
    first_new = len(G.INPUT_FACESETS)
    messages = []
    for k, path in enumerate(paths):
        if progress is not None:
            progress(k / max(len(paths), 1), desc='Loading source faces')
        name = os.path.basename(path)
        try:
            messages.append(_add_source_file(path, name, extract_face_images, get_image_frame))
        except Exception as e:
            # a damaged photo or faceset: the others still load, and the
            # gallery shows what did (it used to stop at the bad file with
            # the earlier ones added but not shown)
            print(f'[source] {name}: {e}')
            messages.append(f'{name}: cannot be read ({e}), skipped')
    if len(G.INPUT_FACESETS) > first_new:
        active_source = first_new          # the newest source is the one in use
    return messages


def _add_source_file(path, name, extract_face_images, get_image_frame):
    """One photo (every face becomes a source) or .fsz (one source); the
    message for it."""
    if path.lower().endswith('.fsz'):
        face_set, caption, msg = _load_faceset(path)
        if face_set is None:
            return msg
        face_set.files = [path]
        G.INPUT_FACESETS.append(face_set)
        ui.globals.ui_input_thumbs.append(face_set.thumb)
        source_captions.append(caption)
        return msg or f'{name}: faceset added ({caption.split(" · ")[-1]})'
    if util.has_image_extension(path):
        found = extract_face_images(path, (False, 0))
        if not found:
            return f'{name}: no face found, not added (try a lower Min. confidence under Detection & tracking)'
        image = get_image_frame(path)
        for i, (face, crop) in enumerate(found):
            face.mask_offsets = (0, 0, 0, 0, 1, 20)
            face_set = FaceSet()
            face_set.faces.append(face)
            face_set.ref_images.append(image)
            face_set.from_photo = True
            face_set.files = [path]
            G.INPUT_FACESETS.append(face_set)
            ui.globals.ui_input_thumbs.append(util.convert_to_gradio(crop))
            source_captions.append(name if len(found) == 1 else f'{name} · face {i + 1}')
        return (f'{name}: 1 face added' if len(found) == 1 else
                f'{name}: {len(found)} faces added as {len(found)} sources')
    return f'{name}: not a photo or faceset, skipped'


def path_start():
    """What the path boxes start with: the Settings folder (path_start, Colab's
    Drive folder by default) when it exists here, else nothing."""
    folder = os.path.expanduser(str(getattr(G.CFG, 'path_start', '') or '').strip())
    if not folder or not os.path.isdir(folder):
        return ''
    return folder if folder.endswith(tuple(filter(None, (os.sep, os.altsep)))) else folder + os.sep


def _only_path_start(path):
    """The box still holds just the start folder (by default also the output
    folder): Add / Enter must not add everything in it."""
    start = path_start()
    return bool(start) and os.path.normpath(path) == os.path.normpath(start)


def add_source_path(path):
    """A .fsz / photo file, or a folder of them (not recursive)."""
    path = (path or '').strip()
    if not path:
        return ['Type the path of a faceset, a photo or a folder']
    if _only_path_start(path):
        return [f'Type the name of the faceset, photo or folder after {path_start()}']
    if os.path.isfile(path):
        return add_sources([path]) if is_source_file(path) else [f'Not a photo or faceset: {path}']
    if os.path.isdir(path):
        files = sorted(p for p in (os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.'))
                       if os.path.isfile(p) and is_source_file(p))
        return add_sources(files) if files else [f'No photos or facesets in {path}']
    return [f'Not found: {path}']


def _load_faceset(path):
    """One FaceSet from a .fsz (all faces of all its PNGs, averaged), its
    caption, and a message (faceset check or error)."""
    from unleashed.face_util import extract_face_images
    from unleashed.capturer import get_image_frame
    name = os.path.basename(path)
    folder = tempfile.mkdtemp(prefix='faceset_')
    try:
        try:
            util.unzip(path, folder)
        except Exception as e:
            # truncated (a save cut off by a disconnect), not a zip, bad CRC ...
            print(f'[faceset] {name}: {e}')
            return None, None, f'{name}: damaged or not a faceset file, not added'
        face_set = FaceSet()
        main_faces = []
        thumb = None
        # os.listdir order, as before: faces[0] holds the averaged identity, and
        # summing in another order could change it in the last float bits
        for file in os.listdir(folder):
            if not file.endswith('.png'):
                continue
            filename = os.path.join(folder, file)
            found = extract_face_images(filename, (False, 0))
            if found:
                areas = [(f[0].bbox[2] - f[0].bbox[0]) * (f[0].bbox[3] - f[0].bbox[1]) for f in found]
                main_faces.append(len(face_set.faces) + int(np.argmax(areas)))
            for face, crop in found:
                face.mask_offsets = (0, 0, 0, 0, 1, 20)
                face_set.faces.append(face)
                if thumb is None:
                    thumb = util.convert_to_gradio(crop)
                face_set.ref_images.append(get_image_frame(filename))
        if not face_set.faces:
            return None, None, f'{name}: no face found in the faceset'
        msg = None
        flagged = 0
        if len(main_faces) > 2:
            try:
                from unleashed import faceset_check
                refs = face_set.ref_images
                rows = faceset_check.check([faceset_check.photo_metrics(face_set.faces[i], refs[i], geometry=False)
                                            for i in main_faces])
                text = faceset_check.summary(rows, shape=False)
                print(f'[faceset] {name}: {text}')
                flagged = sum(1 for r in rows if r['removable'])
                if flagged:
                    msg = f'{name}: {text}. Review it in the Face Management tab.'
            except Exception as e:
                print(f'[faceset] check skipped ({e})')
        if len(face_set.faces) > 1:
            face_set.AverageEmbeddings()
        face_set.thumb = thumb
        caption = f'{name} · {len(main_faces)} photos' + (f' · {flagged} flagged' if flagged else '')
        return face_set, caption, msg
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def select_source(index):
    global active_source
    if 0 <= index < len(G.INPUT_FACESETS):
        active_source = index


def _take_sources(indexes):
    """Remove these sources (their records, ascending position)."""
    records = []
    for i in sorted(set(indexes), reverse=True):
        records.append((i, G.INPUT_FACESETS.pop(i), ui.globals.ui_input_thumbs.pop(i), source_captions.pop(i)))
    return records[::-1]


def remove_source(index):
    """The × on a source's thumbnail."""
    global active_source
    _sync_sources()
    if not 0 <= index < len(G.INPUT_FACESETS):
        return False
    in_use = G.INPUT_FACESETS[active_source_index()]
    _take_sources([index])
    active_source = next((i for i, s in enumerate(G.INPUT_FACESETS) if s is in_use), min(index, len(G.INPUT_FACESETS) - 1))
    active_source = max(active_source, 0)
    return True


def _copy_face(face):
    """A separate insightface Face with the same values (copy.copy fails on it:
    its __getattr__ answers None for __setstate__)."""
    c = face.__class__()
    dict.update(c, face)
    c.__dict__.update(face.__dict__)
    return c


def _unit_embedding(face_set):
    e = np.asarray(face_set.faces[0].embedding, dtype=np.float32)
    return e / max(float(np.linalg.norm(e)), 1e-6)


def combine_photo_sources():
    """Blend the photo sources of the person in use into one faceset (like a
    .fsz: one identity averaged over the photos). Faces of other people (a
    group photo) and loaded .fsz facesets stay as they are."""
    global active_source
    same = same_person_photos()
    if len(same) < 2:
        return 'No other photo of the person in use: nothing to combine'
    a = active_source_index()
    photos = [i for i, s in enumerate(G.INPUT_FACESETS) if getattr(s, 'from_photo', False)]
    combined = FaceSet()
    combined.files = [f for i in same for f in getattr(G.INPUT_FACESETS[i], 'files', [])]
    for i in same:
        # copies: AverageEmbeddings writes the average into faces[0], which
        # must not change the photo's own Face object
        combined.faces.extend(_copy_face(f) for f in G.INPUT_FACESETS[i].faces)
        combined.ref_images.extend(G.INPUT_FACESETS[i].ref_images)
    combined.AverageEmbeddings()
    thumb = ui.globals.ui_input_thumbs[a]
    position = same[0]
    _take_sources(same)
    G.INPUT_FACESETS.insert(position, combined)
    ui.globals.ui_input_thumbs.insert(position, thumb)
    source_captions.insert(position, f'{len(same)} photos combined')
    active_source = position
    others = len(photos) - len(same)
    return (f'{len(same)} photos of the person in use combined into one faceset'
            + (f'; {others} face{"s" if others != 1 else ""} of other people left as {"they are" if others != 1 else "it is"}'
               if others else ''))


def shuffle_sources():
    """Random order for 'One source per face' (drawn once, shown in the
    numbered captions, used by preview and render alike)."""
    global active_source
    _sync_sources()
    order = list(range(len(G.INPUT_FACESETS)))
    random.shuffle(order)
    G.INPUT_FACESETS[:] = [G.INPUT_FACESETS[i] for i in order]
    ui.globals.ui_input_thumbs[:] = [ui.globals.ui_input_thumbs[i] for i in order]
    source_captions[:] = [source_captions[i] for i in order]
    active_source = order.index(active_source) if active_source < len(order) else 0


# ----------------------------------------------------------------------------- targets

targets = []                          # see add_targets
selected_target = None
_next_target = 0


def _probe(path):
    """(kind, frames, fps, first frame BGR) of a target file."""
    if util.has_image_extension(path):
        from unleashed.capturer import get_image_frame
        return 'image', 1, 0.0, get_image_frame(path)
    cap = cv2.VideoCapture(path)
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
    finally:
        cap.release()
    fps = float(util.detect_fps(path) or 0) if frames > 0 else 0.0
    return ('gif' if path.lower().endswith('.gif') else 'video'), frames, fps, frame if ok else None


def _variable_frame_rate(path):
    """True when ffprobe reports an average frame rate that differs from the
    stream's nominal rate by more than 1.5 % (phone clips that slow down in low
    light, screen recordings). The in-memory render times frames at the
    average rate, so the sound drifts from the picture in such a video."""
    import shutil as _sh
    import subprocess
    if _sh.which('ffprobe') is None:
        return False
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                              'stream=r_frame_rate,avg_frame_rate', '-of', 'json', path],
                             capture_output=True, text=True, timeout=20).stdout
        stream = (json.loads(out or '{}').get('streams') or [{}])[0]

        def rate(text):
            num, _, den = str(text or '').partition('/')
            return float(num) / float(den or 1) if float(den or 1) else 0.0

        nominal, average = rate(stream.get('r_frame_rate')), rate(stream.get('avg_frame_rate'))
        if nominal > 0 and average > 0:
            return abs(nominal - average) / max(nominal, average) > 0.015
    except Exception:
        pass
    return False


def _serve_in_place(path):
    """A target added by its path (Drive, a local folder) is listed in the
    file list, and Gradio copies any listed file that is not in its upload
    cache into that cache (a full copy of a large video, hashed again at every
    refresh). Serving it as a static file lists it where it is."""
    temp = os.environ.get('GRADIO_TEMP_DIR') or os.environ.get('TEMP') or ''
    if temp and os.path.abspath(path).startswith(os.path.abspath(temp) + os.sep):
        return                                # uploaded: already in Gradio's cache
    import gradio as gr
    gr.set_static_paths([path])


def add_targets(paths, progress=None):
    """Add target files (images, videos, GIFs); a path already in the list is
    not added again. Returns per-file messages for files that were not added."""
    global _next_target, selected_target
    from unleashed.capturer import prepare_seek_copy
    known = {t['path'] for t in targets}
    messages = []
    for k, path in enumerate(paths):
        if progress is not None:
            progress(k / max(len(paths), 1), desc='Adding target files')
        name = os.path.basename(path)
        if path in known:
            continue
        if not os.path.isfile(path):
            messages.append(f'{name}: not found, not added')
            continue
        if not is_target_file(path):
            messages.append(f'{name}: not an image, video or GIF, skipped')
            continue
        kind, frames, fps, first = _probe(path)
        if kind != 'image' and frames < 1:
            messages.append(f'{name}: its frames cannot be read, skipped')
            continue
        _next_target += 1
        t = {'id': _next_target, 'path': path, 'name': name, 'kind': kind, 'frames': frames, 'fps': fps,
             'start': 1, 'end': frames, 'out_fps': 0.0, 'mask': None, 'mask_frame': None,
             'vfr': kind == 'video' and _variable_frame_rate(path)}
        _serve_in_place(path)
        targets.append(t)
        known.add(path)
        if kind == 'video':
            prepare_seek_copy(path)
        if selected_target is None or target() is None:
            selected_target = t['id']
    return messages


def add_target_path(path):
    """A file, or every image / video in a folder (not recursive). A wrong
    path changes nothing."""
    path = (path or '').strip()
    if not path:
        return ['Type the path of a file or a folder']
    if _only_path_start(path):
        return [f'Type the name of the file or folder after {path_start()}']
    if os.path.isfile(path):
        return add_targets([path])
    if os.path.isdir(path):
        # full paths: util.is_video only accepts a file that exists
        files = sorted(p for p in (os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.'))
                       if os.path.isfile(p) and is_target_file(p))
        return add_targets(files) if files else [f'No images or videos in {path}']
    return [f'Not found: {path}']


def target(tid=None):
    tid = selected_target if tid is None else tid
    return next((t for t in targets if t['id'] == tid), None)


def target_paths():
    """The file list's value (the targets, in order)."""
    return [t['path'] for t in targets]


def selected_target_index():
    return next((i for i, t in enumerate(targets) if t['id'] == selected_target), None)


def select_target(index):
    global selected_target
    if 0 <= index < len(targets):
        selected_target = targets[index]['id']


def remove_target(path):
    """The × of a file in the list."""
    global selected_target
    i = next((k for k, t in enumerate(targets) if t['path'] == path), None)
    if i is None:
        return False
    removed = targets.pop(i)
    if removed['id'] == selected_target:
        selected_target = targets[min(i, len(targets) - 1)]['id'] if targets else None
    return True


def clear_targets():
    global selected_target
    targets.clear()
    selected_target = None


def clock(frame, fps):
    """Start time of frame `frame` (1-based), short: 0:01.25 / 1:02:03.50."""
    if not fps:
        return ''
    cs = int(round(max(0.0, (frame - 1) / fps) * 100))         # centiseconds: 59.995 s is 1:00.00, not 0:60.00
    h, rest = divmod(cs, 360000)
    m, rest = divmod(rest, 6000)
    sec = f'{rest // 100:02d}.{rest % 100:02d}'
    return f'{h}:{m:02d}:{sec}' if h else f'{m}:{sec}'


def duration(frames, fps):
    """Length of `frames` frames: 0:07 / 1:02:03."""
    if not fps:
        return ''
    secs = int(round(frames / fps))
    m, sec = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m}:{sec:02d}'


def set_range(which, frame):
    """Start or end of the selected video at the frame shown (1-based,
    inclusive). Returns a note when the other end had to move."""
    t = target()
    if t is None or t['kind'] == 'image':
        return 'Select a video first'
    frame = max(1, min(int(frame), t['frames']))
    if which == 'start':
        t['start'] = frame
        if t['end'] < frame:
            t['end'] = t['frames']
            return 'The end was before this frame, so it moved to the end of the video.'
    else:
        t['end'] = frame
        if t['start'] > frame:
            t['start'] = 1
            return 'The start was after this frame, so it moved to the first frame.'
    return None


def reset_range():
    t = target()
    if t is not None and t['kind'] != 'image':
        t['start'], t['end'] = 1, t['frames']


def range_text():
    t = target()
    if t is None or t['kind'] == 'image':
        return ''
    whole = (t['start'], t['end']) == (1, t['frames'])
    n = t['end'] - t['start'] + 1
    if whole:
        length = clock(t['frames'] + 1, t['fps'])
        return f"Render: whole video · {t['frames']} frames" + (f" · {length}" if length else '')
    span = f"{clock(t['start'], t['fps'])}–{clock(t['end'] + 1, t['fps'])}" if t['fps'] else ''
    return f"**Render {t['start']}–{t['end']}** · {n} frames" + (f" · {span}" if span else '')


def set_out_fps(value):
    t = target()
    if t is not None and t['kind'] != 'image':
        t['out_fps'] = max(0.0, float(value or 0))


def process_entries():
    """The render list: fresh entries, the frame range converted to the
    engine's 0-based [start, end), the painted mask attached to its file."""
    entries = []
    for t in targets:
        if t['kind'] == 'image':
            e = ProcessEntry(t['path'], 0, 0, 0)
        else:
            e = ProcessEntry(t['path'], t['start'] - 1, t['end'], float(t['out_fps'] or 0))
        e.imagemask = t['mask']
        entries.append(e)
    return entries


# ----------------------------------------------------------------------------- people (Specific people)



def people_gallery():
    return list(ui.globals.ui_target_thumbs)


def _unit(emb):
    e = np.asarray(emb, np.float64).reshape(-1)
    return e / (np.linalg.norm(e) + 1e-12)


def add_person(face, crop):
    """Add a person to replace; the same person (embedding distance < 0.2)
    is not added twice. Returns a message."""
    e = _unit(face.embedding)
    for existing in G.TARGET_FACES:
        if 1.0 - float(e @ _unit(existing.embedding)) < 0.2:
            return 'This person is already in the list'
    G.TARGET_FACES.append(face)
    ui.globals.ui_target_thumbs.append(util.convert_to_gradio(crop))
    return None


def remove_person(index):
    """The × on a picked person's thumbnail."""
    if not 0 <= index < len(G.TARGET_FACES):
        return False
    G.TARGET_FACES.pop(index)
    ui.globals.ui_target_thumbs.pop(index)
    return True


def faces_in_frame(frame_num):
    """([(face, crop)], problem) for the frame the preview shows, found as
    the swap finds them (rotated faces too), left to right."""
    from unleashed.face_util import extract_face_images
    t = target()
    if t is None:
        return [], 'Add a target file first'
    if not os.path.isfile(t['path']):
        return [], f"{t['name']} is gone (moved or deleted): add it again"
    if t['kind'] == 'image':
        found = extract_face_images(t['path'], (False, 0), use_multi_angle=True)
    else:
        found = extract_face_images(t['path'], (True, int(frame_num or 1)), use_multi_angle=True)
    if not found:
        return [], 'No face found in this frame'
    return found, None


# ----------------------------------------------------------------------------- manual mask (per target file)

def set_target_mask(editor_value, frame_num, tid=None):
    """Keep the painting (editor dict) on the target it was painted on (tid;
    default the selected one). Returns True when something was painted."""
    t = target(tid)
    if t is None:
        return False
    layers = (editor_value or {}).get('layers') or []
    layer = layers[0] if layers else None
    from unleashed.ProcessMgr import ProcessMgr
    if layer is not None and ProcessMgr.manual_mask(layer) is None:
        layer = None
    t['mask'] = layer
    t['mask_frame'] = int(frame_num) if layer is not None else None
    return layer is not None


def clear_target_mask():
    t = target()
    if t is not None:
        t['mask'], t['mask_frame'] = None, None


def mask_info():
    t = target()
    if t is None:
        return ''
    if t['mask'] is None:
        return ''
    if t['kind'] == 'image':
        return 'Painted areas stay original'
    return f"Painted on frame {t['mask_frame']} · used on every frame"


# ----------------------------------------------------------------------------- run checks

run_lock = threading.Lock()


def readiness():
    """(ready, markdown): what Start will do, or what is missing, plus warnings."""
    problems = []
    if not G.INPUT_FACESETS:
        problems.append('add a source face')
    if not targets:
        problems.append('add a target file')
    mode = values['mode']
    if MODES[mode] == 'selected' and not G.TARGET_FACES:
        problems.append('pick the faces with "Use face from the frame"')
    if problems:
        return False, 'To start: ' + ', '.join(problems) + '.'
    images = [t for t in targets if t['kind'] == 'image']
    videos = [t for t in targets if t['kind'] != 'image']
    parts = []
    if images:
        parts.append(f"{len(images)} image{'s' if len(images) != 1 else ''}")
    if videos:
        frames = sum(t['end'] - t['start'] + 1 for t in videos)
        secs = sum((t['end'] - t['start'] + 1) / t['fps'] for t in videos if t['fps'])
        length = f'{secs:.1f} s' if secs < 10 else duration(secs, 1.0)
        parts.append(f"{len(videos)} video{'s' if len(videos) != 1 else ''} ({frames} frames, {length})")
    who = mode
    if MODES[mode] == 'all_input':
        who += f' ({len(G.INPUT_FACESETS)} sources)'
    elif MODES[mode] == 'selected':
        who += f' ({len(G.TARGET_FACES)} picked) ← {source_captions[active_source_index()]}'
    else:
        who += f' ← {source_captions[active_source_index()]}'
    text = f"**Ready:** {' + '.join(parts)} · {who}"
    warnings = run_warnings()
    if warnings:
        text += '\n\n' + '\n'.join(f'⚠ {w}' for w in warnings)
    return True, text


def run_warnings():
    v = values
    w = []
    if v['enhancer'] != 'None' and int(str(v['resolution'])[:-2]) > 512:
        w.append('The enhancer works at 512px: a higher resolution only costs time.')
    if NO_FACE[v['no_face']] == 2 and v['method'] == METHOD_MEMORY and any(t['kind'] != 'image' for t in targets):
        w.append('"Drop the frame": the video gets shorter and the sound drifts.')
    if MODES[v['mode']] == 'all_input' and len(G.INPUT_FACESETS) < 2:
        w.append('"One source per face" with one source: only the leftmost face is swapped.')
    if v['er'] and ((v['keep_mouth'] and v['er_mouth']) or (v['keep_eyes'] and v['er_eyes'])):
        w.append("Pasted-back mouth / eyes override the restored expression there.")
    names = [os.path.splitext(t['name'])[0] for t in targets]
    if len(names) != len(set(names)):
        w.append('Same file names: results get _2, _3 …')
    if G.CFG.clear_output:
        w.append('Start deletes the results of earlier runs from the output folder (Settings).')
    vfr = [t['name'] for t in targets if t.get('vfr')]
    if vfr:
        w.append(f"{', '.join(vfr[:2])}{' …' if len(vfr) > 2 else ''}: variable frame rate. In memory the sound can "
                 "drift from the picture (up to about a second); render it with Extract frames (whole video), or "
                 "convert it to a constant frame rate first.")
    plugin, warn = mask_plugin()
    if warn:
        w.append(warn)
    return w


_OUTPUTS_FILE = '.unleashed_outputs.json'      # in the output folder: the files renders wrote there


def _outputs_path(folder):
    return os.path.join(folder, _OUTPUTS_FILE)


def _read_outputs(folder):
    """[{'name', 'size', 'mtime'}] of the results recorded in this folder."""
    try:
        with open(_outputs_path(folder), encoding='utf-8') as fh:
            rows = json.load(fh)
        return [r for r in rows if isinstance(r, dict) and isinstance(r.get('name'), str)]
    except Exception:
        return []


def _write_outputs(folder, rows):
    try:
        if rows:
            with open(_outputs_path(folder), 'w', encoding='utf-8') as fh:
                json.dump(rows, fh, indent=0)
        elif os.path.isfile(_outputs_path(folder)):
            os.remove(_outputs_path(folder))
    except OSError as e:
        print(f'[output] could not record the results ({e})')


def _unchanged(path, row):
    """The file is still the result the render wrote (same size and time): a
    file the user saved later under that name is theirs."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_size == row.get('size') and abs(st.st_mtime - float(row.get('mtime') or 0)) < 2


def record_outputs(paths):
    """Remember the files a render wrote (per output folder), so 'Delete
    earlier results' deletes those and nothing else."""
    by_folder = {}
    for p in paths:
        if p and os.path.isfile(p):
            st = os.stat(p)
            by_folder.setdefault(os.path.dirname(os.path.abspath(p)), []).append(
                {'name': os.path.basename(p), 'size': st.st_size, 'mtime': st.st_mtime})
    for folder, rows in by_folder.items():
        names = {r['name'] for r in rows}
        _write_outputs(folder, [r for r in _read_outputs(folder) if r['name'] not in names] + rows)


def _files_in_use():
    """Real paths of the targets and of the photos / facesets the sources came from."""
    used = {os.path.realpath(t['path']) for t in targets}
    for face_set in G.INPUT_FACESETS:
        used.update(os.path.realpath(f) for f in getattr(face_set, 'files', []) or [])
    return used


def clear_output_media():
    """'Delete earlier results at Start': delete the results of earlier runs in
    the output folder: the files renders recorded there (and still as they
    wrote them), and the unfinished __temp parts of stopped renders. Never a
    target or source in use and never anything else (on Colab the output
    folder is also where the user keeps targets and photos: the old version
    deleted every image and video in it, the file being rendered included)."""
    folder = G.output_path
    if not folder or not os.path.isdir(folder):
        return 0
    in_use = _files_in_use()
    rows = _read_outputs(folder)
    recorded = {r['name'] for r in rows}
    leftovers = [f for f in os.listdir(folder)
                 if '__temp' in os.path.splitext(f)[0] and f not in recorded and is_target_file(os.path.join(folder, f))]
    removed, kept = 0, []
    for row in rows + [{'name': f, 'leftover': True} for f in leftovers]:
        p = os.path.join(folder, row['name'])
        if not os.path.isfile(p):
            continue
        if os.path.realpath(p) in in_use or not (row.get('leftover') or _unchanged(p, row)):
            if not row.get('leftover') and os.path.realpath(p) in in_use:
                kept.append(row)                 # a result the user added back as a target
            continue
        try:
            os.remove(p)
            removed += 1
        except OSError:
            if not row.get('leftover'):
                kept.append(row)
    _write_outputs(folder, kept)
    return removed


def drop_missing_targets():
    """Targets whose file is gone (deleted, renamed or moved on Drive; a Clean
    temp) leave the list: Gradio's file list cannot show a missing file, and
    every event returning the list failed. Returns the names removed."""
    global selected_target
    gone = [t['name'] for t in targets if not os.path.isfile(t['path'])]
    if gone:
        targets[:] = [t for t in targets if os.path.isfile(t['path'])]
        if target() is None:
            selected_target = targets[0]['id'] if targets else None
    return gone


def reset_after_temp_clean():
    """Settings > Clean temp folder deleted the uploads. Targets uploaded into
    the temp folder leave the list (their files are gone); targets added by
    path stay (their thumbnails come back on the next refresh). Sources and
    picked people are in memory and stay. Returns the names removed."""
    from unleashed.capturer import prepare_seek_copy
    gone = drop_missing_targets()
    for t in targets:
        if t['kind'] == 'video':
            prepare_seek_copy(t['path'])         # its quick-seek copy was in the temp folder
    return gone
