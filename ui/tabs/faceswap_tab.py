"""Face Swap tab: layout and wiring. The state and logic live in
ui/tabs/faceswap_state.py (S).

Layout, left to right in the order you work (compact for a 1920x1080 screen):
  Source faces · Target files · Faces to replace · Swap (titled boxes)  |
  preview on top, then frame (◀ ▶, arrow keys) · range · view · Ready +
  Start / Stop · status  |  settings in boxes (Expression, Occlusion, Enhance,
  Detection & tracking, Video output), defaults at the bottom. Results are
  not shown here: they are in the output folder the status line names.

How the preview follows the settings: every setting's .change bumps a hidden
counter in the browser (js only, no server call); the counter's .change runs
one preview with the current panel values (always_last, one at a time), so a
burst of changes gives one or two previews. Server-side changes (sources,
targets, people, painting) return the counter + 1 to get the same effect.
Settings are applied at each preview and once at Start, never during a render.
"""
import os
import time
import traceback

import gradio as gr
import numpy as np

import unleashed.globals
import unleashed.utilities as util
from unleashed.capturer import get_image_frame, get_video_frame
from unleashed import colab
from ui.tabs import faceswap_state as S

G = unleashed.globals
INTERNAL = dict(show_api=False)


def _info(message):
    """Short-lived confirmation (the toasts sit over the settings column's top)."""
    gr.Info(message, duration=5)

BUMP_JS = '(n) => (n || 0) + 1'
# ◀ ▶ in the browser, moving the slider itself: Gradio reads a js event's
# inputs asynchronously, so a click that came before the previous one had been
# applied read the same frame and was lost (a server round trip was worse)
STEP_JS = '''(n) => {
    const el = document.querySelector('#frame_slider input[type=range]');
    if (el) {
        const v = Math.max(Number(el.min) || 1, Math.min(Number(el.max), Math.round(Number(el.value)) + %d));
        if (v !== Number(el.value)) {
            el.value = v;
            el.dispatchEvent(new Event('input', {bubbles: true}));
        }
    }
    return (n || 0) + 1;
}'''

settings = {}              # key -> component
C = {}                     # other components by name (for the load / cross-tab refresh)


def _s(key, comp):
    settings[key] = comp
    return comp


def _vals(data):
    """Panel values by key from an event's {component: value} input. An
    emptied number box sends None: an empty crop box means no crop, any other
    keeps the value in use (a slider's box puts its minimum back on blur)."""
    out = {}
    for k, c in settings.items():
        if c in data:
            v = data[c]
            if v is None:
                if not k.startswith('crop_'):
                    continue
                v = 0.0
            out[k] = v
    return out


def _next_tick(tick):
    """The preview tick a server handler returns. tick + 1 could equal the
    value the browser reached meanwhile with its own +1 (one setting changed
    while a slow source/target handler ran): an equal value fires no change,
    so no preview came after the server change. A millisecond clock never
    meets the browser's small steps."""
    return max(int(tick or 0) + 1, int(time.time() * 1000))


def _drop_missing_targets():
    """Targets whose file disappeared leave the list, with a note (Gradio's
    file list fails on a missing file, and with it every event returning it)."""
    gone = S.drop_missing_targets()
    if gone:
        gr.Warning('No longer found, removed from Target files: ' + ', '.join(gone[:3])
                   + (f' and {len(gone) - 3} more' if len(gone) > 3 else ''))


# The source gallery's highlight (the source in use) is set by the server.
# Gradio's Gallery then fires `select` as if it had been clicked, but only
# when its highlight actually changes; processed after a newer action, that
# echo would move the choice back. So the server follows what the gallery
# highlights in the browser (Gallery.svelte: a new value keeps the highlight
# while it is in range and drops it after the gallery was empty) and notes an
# echo only for a highlight that really changes. A real click on another
# thumbnail clears the notes; notes also expire.
_shown = {'src': None}
_emptied = {'src': True}
_echo = {'src': []}
ECHO_SECONDS = 30


def _gallery(kind, items):
    """gr.Gallery(value=items), following what the browser will highlight."""
    n = len(items)
    if n == 0:
        _shown[kind], _emptied[kind] = None, True
    elif _emptied[kind]:
        _shown[kind], _emptied[kind] = None, False
    elif _shown[kind] is not None and _shown[kind] >= n:
        _shown[kind] = None
    return gr.Gallery(value=items)


def _page_loaded():
    """A new page: its galleries start empty-handed (nothing highlighted) and
    no editor is open."""
    for kind in _shown:
        _shown[kind], _emptied[kind] = None, True
        _echo[kind] = []
    _painting.update(tid=None, frame=None)


def _highlight(kind, index):
    if index is not None and index != _shown[kind]:
        _echo[kind] = [e for e in _echo[kind] if time.time() - e[1] < ECHO_SECONDS][-3:] + [(index, time.time())]
    _shown[kind] = index
    return gr.Gallery(selected_index=index)


def _is_echo(kind, index):
    _shown[kind] = index                        # the browser highlights it now
    notes = [e for e in _echo[kind] if time.time() - e[1] < ECHO_SECONDS]
    hit = next((e for e in notes if e[0] == index), None)
    if hit is not None:
        notes.remove(hit)
        _echo[kind] = notes
        return True
    _echo[kind] = []
    return False


def src_highlight():
    return _highlight('src', S.active_source_index() if G.INPUT_FACESETS else None)


def _title(name, key):
    extra = S.summary(key)
    return f'{name} · {extra}' if extra else name


# ============================================================================ layout

_built = False


def faceswap_tab():
    global _built
    if not _built:
        # the saved defaults once per app start: Settings > Restart Server
        # builds the tab again and must keep the panel as it is
        S.load_saved_defaults()
        _built = True
    gone = S.drop_missing_targets()           # a missing file would fail the list's build
    if gone:
        print(f'[targets] no longer found, removed from the list: {", ".join(gone)}')
    V = S.values
    video = (S.target() or {}).get('kind') in ('video', 'gif')
    t = S.target()

    with gr.Tab("Face Swap"):
        # rendered but hidden: the browser compares it with the tick the last
        # preview answered (preview_done) to show "Updating…" / "Updated"
        tick = gr.Number(value=0, elem_id="fs_tick", elem_classes="fs-hidden", container=False)
        C['tick'] = tick
        C['preview_done'] = gr.Textbox(elem_id="fs_preview_done", elem_classes="fs-hidden", show_label=False,
                                       container=False)
        with gr.Row(elem_id="swap_row", equal_height=False):

            # --------------------------------------------------------------- left: set up
            with gr.Column(scale=3, min_width=280, elem_id="fs_left"):
                with gr.Accordion(_title("Source faces", 'sources'), open=True, elem_classes="fs-box fs-first") as acc_src:
                    C['acc_src'] = acc_src
                    # whole faces, no captions; the × on each and the order
                    # numbers ("One source per face") come from ui/theme.py
                    C['src_gal'] = gr.Gallery(value=S.sources_gallery(), show_label=False, columns=4, allow_preview=False,
                                              preview=False, interactive=False, object_fit="contain", height="150px",
                                              elem_id="src_gal")
                    C['src_x'] = gr.Textbox(elem_id="src_x", elem_classes="fs-hidden", show_label=False, container=False)
                    with gr.Row(elem_classes="fs-buttons"):
                        C['btn_src_combine'] = gr.Button(S.combine_label(), size="sm", visible=len(S.same_person_photos()) >= 2)
                        C['btn_src_shuffle'] = gr.Button("Shuffle order", size="sm",
                                                         visible=S.MODES[V['mode']] == 'all_input' and len(G.INPUT_FACESETS) > 1)
                    C['src_drop'] = gr.Files(show_label=False, file_count="multiple", file_types=None, elem_id="src_drop")
                    with gr.Row(equal_height=True, elem_classes="fs-path"):
                        C['src_path'] = gr.Textbox(value=S.path_start(), show_label=False, container=False, scale=5,
                                                   max_lines=1, placeholder="or a path: .fsz, photo or folder (Enter)",
                                                   elem_id="src_path")
                        C['btn_src_path'] = gr.Button("Add", size="sm", scale=1, min_width=60)

                with gr.Accordion(_title("Target files", 'targets'), open=True, elem_classes="fs-box") as acc_tgt:
                    C['acc_tgt'] = acc_tgt
                    # the list itself: drop files on it, ↑ adds more, × removes
                    # one, click a name to show it in the preview
                    C['tgt_files'] = gr.Files(value=S.target_paths() or None, show_label=False, file_count="multiple",
                                              file_types=None, height=170, elem_id="tgt_files")
                    C['tgt_shown'] = gr.Textbox(value=(t or {}).get('name', ''), elem_id="tgt_shown", elem_classes="fs-hidden",
                                                show_label=False, container=False)
                    with gr.Row(equal_height=True, elem_classes="fs-path"):
                        C['tgt_path'] = gr.Textbox(value=S.path_start(), show_label=False, container=False, scale=5,
                                                   max_lines=1, placeholder="or a path: file or folder (Enter)",
                                                   elem_id="tgt_path")
                        C['btn_tgt_path'] = gr.Button("Add", size="sm", scale=1, min_width=60)

                with gr.Accordion(_title("Faces to replace", 'faces'), open=True, elem_classes="fs-box") as acc_faces:
                    C['acc_faces'] = acc_faces
                    with gr.Row(equal_height=True):
                        mode = _s('mode', gr.Dropdown(list(S.MODES), value=V['mode'], show_label=False, container=False,
                                                      scale=3, elem_id="mode_dd"))
                        # one face in the frame: it is added right away; several: pick one below
                        C['btn_use_face'] = gr.Button("Use face from the frame", size="sm", scale=2, min_width=150)
                    with gr.Column(visible=False, elem_id="picker_col") as picker_col:
                        C['picker_col'] = picker_col
                        with gr.Row(equal_height=True):
                            gr.Markdown("Several faces in this frame: click the one to replace.", elem_classes="fs-line")
                            C['btn_pick_cancel'] = gr.Button("Cancel", size="sm", scale=0, min_width=70)
                        C['picker_gal'] = gr.Gallery(show_label=False, columns=6, allow_preview=False, preview=False,
                                                     interactive=False, object_fit="contain", height="100px",
                                                     elem_id="picker_gal")
                    with gr.Column(visible=S.MODES[V['mode']] == 'selected', elem_id="people_col") as people_col:
                        C['people_col'] = people_col
                        C['people_gal'] = gr.Gallery(value=S.people_gallery(), show_label=False, columns=5, allow_preview=False,
                                                     preview=False, interactive=False, object_fit="contain", height="100px",
                                                     visible=bool(G.TARGET_FACES), elem_id="people_gal")
                        C['people_x'] = gr.Textbox(elem_id="people_x", elem_classes="fs-hidden", show_label=False, container=False)
                        _s('tolerance', gr.Slider(0.01, 1.0, value=V['tolerance'], step=0.01,
                                                  label="Match tolerance (higher = looser)"))

                with gr.Accordion(f"Swap · {S.summary('swap')}", open=True, elem_classes="fs-box fs-last") as acc_swap:
                    with gr.Row():
                        _s('resolution', gr.Dropdown(S.RESOLUTIONS, value=V['resolution'], label="Resolution"))
                        C['passes'] = _s('passes', gr.Slider(1, 5, value=V['passes'], step=1, label="Passes"))
                    with gr.Row(visible=int(V['passes']) > 1) as passes_keep_row:
                        C['passes_keep_row'] = passes_keep_row
                        _s('passes_keep', gr.Checkbox(value=V['passes_keep'],
                                                      label="Keep the first pass's size and colour"))
                    with gr.Row():
                        _s('identity', gr.Slider(0.0, 1.0, value=V['identity'], step=0.05, label="Identity boost"))
                        _s('face_shape', gr.Slider(0.0, 1.0, value=V['face_shape'], step=0.05, label="Source jaw & chin"))

            # --------------------------------------------------------------- centre: preview on top, controls below
            with gr.Column(scale=5, min_width=460, elem_id="center_stage", elem_classes="fs-box fs-first fs-last"):
                C['preview'] = gr.Image(label="Preview", show_label=False, interactive=False, format="jpeg", height="56vh",
                                        show_download_button=False, show_fullscreen_button=True, elem_id="preview_img")
                with gr.Column(visible=False) as editor_col:
                    C['editor_col'] = editor_col
                    C['editor'] = gr.ImageEditor(type="numpy", sources=(), transforms=(), layers=False, height="56vh",
                                                 brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", color_mode="fixed", default_size=40),
                                                 eraser=gr.Eraser(default_size=40), show_download_button=False,
                                                 label="Paint in white what stays original (all frames), then Done",
                                                 elem_id="mask_editor")
                    with gr.Row(elem_classes="fs-buttons"):
                        C['btn_paint_done'] = gr.Button("Done", variant="primary", size="sm")
                        C['btn_paint_clear'] = gr.Button("Clear", size="sm")
                        C['btn_paint_cancel'] = gr.Button("Cancel", size="sm")
                # the timeline right under the picture, like a player
                with gr.Row(equal_height=True, visible=video, elem_id="frame_bar") as frame_row:
                    C['frame_row'] = frame_row
                    C['btn_prev'] = gr.Button("◀", size="sm", scale=0, min_width=40, elem_id="frame_prev")
                    C['frame'] = gr.Slider(1, max(2, (t or {}).get('frames', 2)), value=1, step=1, label="Frame  (← → keys)",
                                           scale=6, elem_id="frame_slider")
                    C['btn_next'] = gr.Button("▶", size="sm", scale=0, min_width=40, elem_id="frame_next")
                with gr.Row(equal_height=True, visible=video, elem_id="range_bar") as range_row:
                    C['range_row'] = range_row
                    with gr.Row(elem_classes="fs-seg", elem_id="range_buttons"):
                        C['btn_start_here'] = gr.Button("From here", size="sm", scale=0, min_width=90)
                        C['btn_end_here'] = gr.Button("To here", size="sm", scale=0, min_width=80)
                        C['btn_whole'] = gr.Button("Whole video", size="sm", scale=0, min_width=100)
                    C['range_md'] = gr.Markdown(S.range_text(), elem_classes="fs-line", elem_id="range_line")
                with gr.Row(equal_height=True, elem_id="view_bar") as view_bar:
                    C['view_bar'] = view_bar
                    C['view'] = gr.Radio(S.VIEWS, value="Swapped", show_label=False, container=False, scale=6, elem_id="view_radio")
                    C['auto'] = gr.Checkbox(value=True, label="Auto-update", container=False, scale=0, min_width=120,
                                          elem_id="fs_auto")
                    C['btn_refresh'] = gr.Button("Refresh", size="sm", scale=0, min_width=80, elem_id="fs_refresh")
                with gr.Row(equal_height=True, elem_id="run_bar"):
                    C['ready_md'] = gr.Markdown(S.readiness()[1], elem_id="ready_line")
                    # only when the notebook cell that started the app can disconnect
                    # (unleashed/colab.py); off at every start
                    C['disconnect'] = gr.Checkbox(value=False, label="Disconnect Colab when done", container=False,
                                                  scale=0, min_width=220, visible=colab.available(),
                                                  elem_id="fs_disconnect")
                    C['btn_start'] = gr.Button("▶ Start", variant="primary", scale=0, min_width=120)
                    C['btn_stop'] = gr.Button("⏹ Stop", variant="secondary", scale=0, min_width=90,
                                              interactive=S.run_lock.locked())
                    C['btn_keep'] = gr.Button("Stay connected", variant="secondary", scale=0, min_width=140,
                                              visible=colab.pending())
                C['status_md'] = gr.Markdown("", elem_id="status_line")

            # --------------------------------------------------------------- right: settings
            with gr.Column(scale=3, min_width=300, elem_id="fs_settings"):
                with gr.Accordion(f"Expression · {S.summary('expression')}", open=True, elem_classes="fs-box fs-first") as acc_expr:
                    er = _s('er', gr.Checkbox(value=V['er'], label="Restore target expression (LivePortrait)"))
                    with gr.Column(visible=V['er']) as er_col:
                        with gr.Row():
                            _s('er_strength', gr.Slider(0, 500, value=V['er_strength'], step=5, label="Strength %"))
                            _s('er_smoothing', gr.Slider(0.0, 1.0, value=V['er_smoothing'], step=0.05, label="Video smoothing"))
                        with gr.Row(elem_id="expr_checks"):
                            _s('er_eyes', gr.Checkbox(value=V['er_eyes'], label="Eyes / blink"))
                            _s('er_mouth', gr.Checkbox(value=V['er_mouth'], label="Mouth"))
                            _s('er_brows', gr.Checkbox(value=V['er_brows'], label="Brows"))
                        with gr.Row(elem_classes="fs-checks"):
                            _s('er_pose_lock', gr.Checkbox(value=V['er_pose_lock'], label="Pose lock (no head drift)"))
                            _s('er_pose_gate', gr.Checkbox(value=V['er_pose_gate'], label="Skip at extreme head angles"))
                        _s('er_structure', gr.Checkbox(value=V['er_structure'], label="Expression only (keep the swapped face shape)"))
                    with gr.Row(elem_classes="fs-checks"):
                        _s('keep_mouth', gr.Checkbox(value=V['keep_mouth'], label="Paste back target's mouth"))
                        _s('keep_eyes', gr.Checkbox(value=V['keep_eyes'], label="Paste back target's eyes"))

                with gr.Accordion(f"Occlusion · {S.summary('occlusion')}", open=False, elem_classes="fs-box") as acc_occ:
                    engine = _s('mask_engine', gr.Dropdown(list(S.MASK_ENGINES), value=V['mask_engine'],
                                                           label="Mask model", show_label=False))
                    with gr.Column(visible=V['mask_engine'] == 'Clip2Seg (by text)') as clip_col:
                        _s('mask_objects', gr.Textbox(value=V['mask_objects'], label="Objects to keep (comma separated)",
                                                      max_lines=1))
                    # the face's edge when a mask model is on (Edge erosion / blur below only
                    # shape the face square around it)
                    with gr.Row(visible=V['mask_engine'] != 'None') as mask_edge_row:
                        C['mask_edge_row'] = mask_edge_row
                        _s('mask_grow', gr.Slider(-10, 10, value=V['mask_grow'], step=0.5, label="Mask grow %"))
                        _s('mask_soften', gr.Slider(0, 10, value=V['mask_soften'], step=0.5, label="Mask soften %"))
                    C['paint_md'] = gr.Markdown(S.mask_info(), visible=bool(S.mask_info()), elem_classes="fs-line")
                    with gr.Row(elem_classes="fs-seg"):
                        C['btn_paint'] = gr.Button("Paint areas that stay original", size="sm")
                        C['btn_paint_remove'] = gr.Button("Remove painting", size="sm",
                                                          interactive=(t or {}).get('mask') is not None)
                    with gr.Row():
                        _s('erosion', gr.Slider(1, 3, value=V['erosion'], step=1, label="Edge erosion"))
                        _s('blur', gr.Slider(4, 100, value=V['blur'], step=1, label="Edge blur"))
                    with gr.Row(elem_classes="fs-checks"):
                        _s('to_chin', gr.Checkbox(value=V['to_chin'], label="Extend swap to chin"))
                        _s('aligned_edges', gr.Checkbox(value=V['aligned_edges'], label="Face-aligned edges"))
                    with gr.Row(elem_id="crop_row"):
                        # no minimum / maximum: Gradio refuses the whole event for a value outside them
                        # (the preview never answered); apply_settings keeps them in 0..0.99
                        _s('crop_top', gr.Number(value=V['crop_top'], step=0.01, precision=2, label="Crop top", min_width=70))
                        _s('crop_bottom', gr.Number(value=V['crop_bottom'], step=0.01, precision=2, label="Bottom", min_width=70))
                        _s('crop_left', gr.Number(value=V['crop_left'], step=0.01, precision=2, label="Left", min_width=70))
                        _s('crop_right', gr.Number(value=V['crop_right'], step=0.01, precision=2, label="Right", min_width=70))
                    _s('color_transfer', gr.Checkbox(value=V['color_transfer'], label="Match colours to target"))

                with gr.Accordion(f"Enhance · {S.summary('enhance')}", open=False, elem_classes="fs-box") as acc_enh:
                    with gr.Row(equal_height=True, elem_id="enh_row"):
                        enh = _s('enhancer', gr.Dropdown(list(S.ENHANCERS), value=V['enhancer'], label="Enhancer",
                                                         show_label=False))
                        with gr.Column(visible=V['enhancer'] != 'None', min_width=160) as enh_col:
                            _s('enhancer_blend', gr.Slider(0.0, 1.0, value=V['enhancer_blend'], step=0.01, label="Strength"))
                    with gr.Column(visible=V['enhancer'] != 'None') as enh_after_col:
                        C['enh_after_col'] = enh_after_col
                        _s('mask_after_enhancer', gr.Checkbox(value=V['mask_after_enhancer'], label="Occlusion mask after enhancer"))

                with gr.Accordion(f"Detection & tracking · {S.summary('detection')}", open=False, elem_classes="fs-box") as acc_det:
                    with gr.Row(elem_classes="fs-checks"):
                        _s('autorotate', gr.Checkbox(value=V['autorotate'], label="Auto-rotate lying faces"))
                        lmk = _s('lmk_align', gr.Checkbox(value=V['lmk_align'], label="68-point alignment"))
                    with gr.Column(visible=V['lmk_align']) as lmk_col:
                        with gr.Row(elem_classes="fs-checks"):
                            _s('hi_lmk', gr.Checkbox(value=V['hi_lmk'], label="2dfan4 landmarks"))
                            C['lmk_gate'] = _s('lmk_gate', gr.Checkbox(value=V['lmk_gate'], label="Landmark sanity gate"))
                        with gr.Column(visible=V['lmk_gate']) as gate_col:
                            C['gate_col'] = gate_col
                            _s('lmk_gate_thr', gr.Slider(0.0, 0.50, value=V['lmk_gate_thr'], step=0.005, label="Gate threshold"))
                    sm = _s('smoothing', gr.Checkbox(value=V['smoothing'], label="Landmark smoothing (video)"))
                    with gr.Column(visible=V['smoothing']) as sm_col:
                        with gr.Row():
                            _s('smoothing_strength', gr.Slider(0.0, 1.0, value=V['smoothing_strength'], step=0.05, label="Strength"))
                            _s('smoothing_deadzone', gr.Slider(0.0, 0.03, value=V['smoothing_deadzone'], step=0.001,
                                                               label="Still-face limit"))
                    C['multi_angle'] = _s('multi_angle', gr.Dropdown(list(S.MULTI_ANGLE), value=V['multi_angle'], label="Rotated faces"))
                    with gr.Column(visible=S.MULTI_ANGLE[V['multi_angle']] == 'always') as upright_col:
                        C['upright_col'] = upright_col
                        _s('upright', gr.Slider(0.0, 1.0, value=V['upright'], step=0.05, label="Upright priority"))
                    with gr.Row():
                        _s('det_thresh', gr.Slider(0.10, 0.90, value=V['det_thresh'], step=0.01, label="Min. confidence"))
                        _s('det_size', gr.Dropdown([320, 640, 1024], value=V['det_size'], label="Detection size"))

                with gr.Accordion(f"Video output · {S.summary('video')}", open=False, elem_classes="fs-box fs-last") as acc_vid:
                    method = _s('method', gr.Radio([(S.METHOD_MEMORY, S.METHOD_MEMORY),
                                                    (f"{S.METHOD_EXTRACT} (no smoothing)", S.METHOD_EXTRACT)],
                                                   value=V['method'], label="Processing"))
                    with gr.Column(visible=V['method'] == S.METHOD_EXTRACT) as keep_col:
                        _s('keep_frames', gr.Checkbox(value=V['keep_frames'], label="Keep extracted frames (until the next start)"))
                    _s('no_face', gr.Dropdown(list(S.NO_FACE), value=V['no_face'], label="When no face gets swapped in a frame"))
                    with gr.Row(equal_height=True):
                        _s('skip_audio', gr.Checkbox(value=V['skip_audio'], label="No audio"))
                        C['out_fps'] = gr.Number(value=(t or {}).get('out_fps', 0), label="Output fps, this video (0 = original)",
                                                 precision=2, minimum=0, visible=video)

                with gr.Row(elem_classes="fs-buttons", elem_id="defaults_bar"):
                    C['btn_save_def'] = gr.Button("Save my defaults", size="sm", min_width=90)
                    C['btn_load_def'] = gr.Button("Load my defaults", size="sm", min_width=90, interactive=S.has_saved_defaults())
                    C['btn_reset_def'] = gr.Button("Factory defaults", size="sm", min_width=90)

        C.update(acc_swap=acc_swap, acc_expr=acc_expr, acc_occ=acc_occ, acc_enh=acc_enh, acc_det=acc_det, acc_vid=acc_vid)

    _wire(tick, er, er_col, engine, clip_col, enh, enh_col, lmk, lmk_col, sm, sm_col, method, keep_col, mode)
    return register_load


# ============================================================================ wiring

def _wire(tick, er, er_col, engine, clip_col, enh, enh_col, lmk, lmk_col, sm, sm_col, method, keep_col, mode):
    # state handlers run one at a time; only the slow ones (adding files) show
    # a progress overlay, which lands on every output (the readiness line too)
    slow = dict(concurrency_id="fs_state", concurrency_limit=1, **INTERNAL)
    one = dict(slow, show_progress="hidden")

    # every setting change refreshes the preview (browser-side bump, coalesced)
    for key, comp in settings.items():
        if key == 'mask_objects':
            comp.submit(None, [tick], [tick], js=BUMP_JS, **INTERNAL)
            comp.blur(None, [tick], [tick], js=BUMP_JS, **INTERNAL)
        else:
            comp.change(None, [tick], [tick], js=BUMP_JS, **INTERNAL)
    for comp in (C['view'], C['auto']):
        comp.change(None, [tick], [tick], js=BUMP_JS, **INTERNAL)
    C['frame'].release(None, [tick], [tick], js=BUMP_JS, **INTERNAL)

    # children shown only when their parent is on (browser only)
    show = lambda cond: f"(v) => ({{__type__: 'update', visible: {cond}}})"
    er.change(None, [er], [er_col], js=show('v'), **INTERNAL)
    engine.change(None, [engine], [clip_col], js=show("v === 'Clip2Seg (by text)'"), **INTERNAL)
    engine.change(None, [engine], [C['mask_edge_row']], js=show("v !== 'None'"), **INTERNAL)
    C['passes'].change(None, [C['passes']], [C['passes_keep_row']], js=show('v > 1'), **INTERNAL)
    enh.change(None, [enh], [enh_col], js=show("v !== 'None'"), **INTERNAL)
    enh.change(None, [enh], [C['enh_after_col']], js=show("v !== 'None'"), **INTERNAL)
    lmk.change(None, [lmk], [lmk_col], js=show('v'), **INTERNAL)
    sm.change(None, [sm], [sm_col], js=show('v'), **INTERNAL)
    method.change(None, [method], [keep_col], js=show(f"v === '{S.METHOD_EXTRACT}'"), **INTERNAL)
    always = next(k for k, v in S.MULTI_ANGLE.items() if v == 'always')
    C['multi_angle'].change(None, [C['multi_angle']], [C['upright_col']], js=show(f"v === '{always}'"), **INTERNAL)
    C['lmk_gate'].change(None, [C['lmk_gate']], [C['gate_col']], js=show('v'), **INTERNAL)

    # preview (it also refreshes the section headers)
    preview_inputs = set(settings.values()) | {C['view'], C['frame'], C['auto'], tick}
    preview_outputs = [C['preview'], C['ready_md']] + [C[k] for k in HEADER_KEYS] + [C['preview_done']]
    tick.change(on_preview, preview_inputs, preview_outputs, trigger_mode="always_last", concurrency_id="fs_preview",
                concurrency_limit=1, show_progress="hidden", **INTERNAL)
    C['btn_refresh'].click(lambda d: on_preview(d, force=True), preview_inputs, preview_outputs, concurrency_id="fs_preview",
                           concurrency_limit=1, show_progress="hidden", **INTERNAL)

    # sources
    det = [settings['det_thresh'], settings['det_size']]
    src_out = [C['src_gal'], C['btn_src_combine'], C['btn_src_shuffle'], C['ready_md'], tick]
    fix_src = dict(fn=src_highlight, inputs=None, outputs=C['src_gal'], show_progress="hidden", **INTERNAL)
    C['src_drop'].upload(on_src_upload, [C['src_drop'], tick] + det, [C['src_drop']] + src_out, **slow).then(**fix_src)
    for ev in (C['btn_src_path'].click, C['src_path'].submit):          # the Add button or Enter
        ev(on_src_path, [C['src_path'], tick] + det, src_out, **slow).then(**fix_src).then(
            _clear_found_path, [C['src_path']], [C['src_path']], show_progress="hidden", **INTERNAL)
    # the mode changes the source numbering (One source per face): keep the highlight
    mode.change(on_mode, [mode], [C['people_col'], C['btn_src_shuffle'], C['ready_md']], **one).then(**fix_src)
    # always_last: a click made while the previous one was pending is not dropped
    C['src_gal'].select(on_src_select, [tick], [C['btn_src_combine'], C['ready_md'], tick],
                        trigger_mode="always_last", **one)
    C['src_x'].input(on_src_x, [C['src_x'], tick], src_out, **one).then(**fix_src)
    C['btn_src_combine'].click(on_src_combine, [tick], src_out, **slow).then(**fix_src)
    C['btn_src_shuffle'].click(on_src_shuffle, [tick], src_out, **one).then(**fix_src)

    # targets: the file list is the list
    tgt_out = [C['tgt_files'], C['tgt_shown'], C['frame_row'], C['range_row'], C['frame'], C['range_md'], C['out_fps'],
               C['paint_md'], C['btn_paint_remove'], C['ready_md'], tick]
    C['tgt_files'].upload(on_tgt_upload, [C['tgt_files'], tick], tgt_out, **slow)
    # the list's remaining files, not gr.DeletedFileData: Gradio refuses event
    # data naming a file outside its upload cache (a target added by path)
    C['tgt_files'].delete(on_tgt_delete, [C['tgt_files'], tick], tgt_out, trigger_mode="always_last", **one)
    C['tgt_files'].clear(on_tgt_clear, [tick], tgt_out, **one)
    C['tgt_files'].select(on_tgt_select, [tick], tgt_out[1:], trigger_mode="always_last",
                          **one)       # not the list: it stays as it is
    for ev in (C['btn_tgt_path'].click, C['tgt_path'].submit):
        ev(on_tgt_path, [C['tgt_path'], tick], tgt_out, **slow).then(
            _clear_found_path, [C['tgt_path']], [C['tgt_path']], show_progress="hidden", **INTERNAL)
    C['btn_prev'].click(None, [tick], [tick], js=STEP_JS % -1, **INTERNAL)
    C['btn_next'].click(None, [tick], [tick], js=STEP_JS % 1, **INTERNAL)
    range_out = [C['range_md'], C['ready_md']]
    C['btn_start_here'].click(lambda f: on_range('start', f), [C['frame']], range_out, **one)
    C['btn_end_here'].click(lambda f: on_range('end', f), [C['frame']], range_out, **one)
    C['btn_whole'].click(lambda f: on_range('whole', f), [C['frame']], range_out, **one)
    C['out_fps'].input(on_out_fps, [C['out_fps']], None, **one)

    # people: "Use face from the frame", the picker for several faces, the × on each
    use_out = [mode, C['picker_col'], C['picker_gal'], C['people_gal'], C['ready_md'], tick]
    C['btn_use_face'].click(on_use_face, [C['frame'], tick], use_out, **slow)
    C['picker_gal'].select(on_pick_face, [tick], use_out, **one)
    C['btn_pick_cancel'].click(on_pick_cancel, None, [C['picker_col'], C['picker_gal']], **one)
    C['people_x'].input(on_person_x, [C['people_x'], tick], [C['people_gal'], C['ready_md'], tick], **one)

    # painted keep-original mask
    paint_out = [C['editor_col'], C['preview'], C['editor'], C['paint_md'], C['btn_paint_remove'], C['acc_occ'],
                 C['view_bar'], C['frame_row'], C['range_row'], tick]
    C['btn_paint'].click(on_paint_open, [C['frame'], tick], paint_out, **one)
    C['btn_paint_done'].click(on_paint_done, [C['editor'], C['frame'], tick], paint_out, **one)
    C['btn_paint_clear'].click(on_paint_clear, [C['frame'], tick], paint_out, **one)
    C['btn_paint_cancel'].click(on_paint_cancel, [tick], paint_out, **one)
    C['btn_paint_remove'].click(on_paint_remove, [tick], paint_out, **one)

    # run
    run_inputs = set(settings.values()) | {tick, C['disconnect']}
    start = C['btn_start'].click(on_start_check, run_inputs, [C['btn_start'], C['btn_stop'], C['status_md']],
                                 show_progress="hidden", **INTERNAL)
    render = start.success(on_render, None, [C['status_md']], show_progress="full", **INTERNAL)
    render.then(on_render_done, [tick], [C['btn_start'], C['btn_stop'], C['ready_md'], tick, C['btn_keep']],
                show_progress="hidden", **INTERNAL)
    C['btn_keep'].click(on_stay_connected, None, [C['btn_keep'], C['status_md']], queue=False, **INTERNAL)
    C['btn_stop'].click(on_stop, None, [C['btn_stop'], C['status_md']], queue=False, **INTERNAL)

    # defaults
    C['btn_save_def'].click(on_save_defaults, set(settings.values()), [C['btn_load_def']], **one)
    keys = list(settings)
    C['btn_load_def'].click(lambda n: on_load_defaults(keys, n), [tick], [settings[k] for k in keys] + [tick], **one)
    C['btn_reset_def'].click(lambda n: on_factory_defaults(keys, n), [tick], [settings[k] for k in keys] + [tick], **one)


def register_load(ui):
    """Page load: show the server's state (sources, targets, people, the last
    applied settings), not the build-time values."""
    keys = list(settings)
    outputs = [settings[k] for k in keys] + refresh_outputs() + [C['src_path'], C['tgt_path'], C['btn_keep']]

    def on_load():
        _page_loaded()
        # the path boxes: only when the start folder changed (Settings) since
        # the page was built, so text typed right after the page opened stays
        start = S.path_start()
        path = start if start != C['src_path'].value else gr.skip()
        return [S.values[k] for k in keys] + refresh_values() + [path, path, gr.Button(visible=colab.pending())]
    ui.load(on_load, None, outputs, show_progress="hidden", **INTERNAL).then(
        src_highlight, None, C['src_gal'], show_progress="hidden", **INTERNAL)


def refresh_outputs():
    """Components refresh_values() fills (also used by other tabs)."""
    return [C['src_gal'], C['btn_src_combine'], C['btn_src_shuffle'],
            C['tgt_files'], C['tgt_shown'], C['frame_row'], C['range_row'], C['frame'], C['range_md'], C['out_fps'],
            C['paint_md'], C['btn_paint_remove'],
            C['people_col'], C['people_gal'],
            C['ready_md'], C['btn_start'], C['btn_stop'], C['tick']]


def refresh_values():
    # Start stays clickable here: this page may be a reload during a render,
    # and only the page that pressed Start hears when it ends (a second Start
    # is refused; a Stop with nothing running says so). The preview tick is a
    # new value (milliseconds): the page's own tick counts up by one.
    _drop_missing_targets()
    return (_src_updates()[:3] + _tgt_updates()[:9] +
            [gr.Column(visible=S.MODES[S.values['mode']] == 'selected')] + _people_updates() +
            [gr.Markdown(S.readiness()[1]), gr.Button(interactive=True), gr.Button(interactive=_rendering()),
             int(time.time() * 1000)])


# ============================================================================ updates

def _src_updates():
    n = len(G.INPUT_FACESETS)
    return [_gallery('src', S.sources_gallery()),
            gr.Button(value=S.combine_label(), visible=len(S.same_person_photos()) >= 2),
            gr.Button(visible=S.MODES[S.values['mode']] == 'all_input' and n > 1),
            gr.Markdown(S.readiness()[1])]


_sent_targets = []             # the target lists last sent to the page's file list, newest last


def _tgt_updates():
    t = S.target()
    paths = S.target_paths()
    if not _sent_targets or _sent_targets[-1] != paths:
        _sent_targets.append(list(paths))
        del _sent_targets[:-10]
    video = t is not None and t['kind'] != 'image'
    frames = t['frames'] if video else 2
    return [gr.Files(value=S.target_paths() or None), gr.Textbox(value=t['name'] if t else ''),
            gr.Row(visible=video), gr.Row(visible=video),
            gr.Slider(minimum=1, maximum=max(2, frames), value=t['start'] if video else 1),
            gr.Markdown(S.range_text()),
            gr.Number(value=t['out_fps'] if video else 0, visible=video),
            _paint_md_update(), _paint_remove_update(),
            gr.Markdown(S.readiness()[1])]


def _people_updates():
    return [gr.Gallery(value=S.people_gallery(), visible=bool(G.TARGET_FACES))]


def _messages(messages):
    problems = [m for m in messages if 'not added' in m or 'skipped' in m or 'not found' in m.lower()
                or 'cannot' in m or 'flagged' in m or 'Review' in m or m.startswith(('Not ', 'No ', 'Type '))]
    if problems:
        gr.Warning('\n'.join(problems[:8]) + (f'\n... and {len(problems) - 8} more' if len(problems) > 8 else ''))
    elif messages:
        _info('\n'.join(messages[:6]) + (f'\n... and {len(messages) - 6} more' if len(messages) > 6 else ''))


def _apply_detection(thresh, size):
    """Source loading and face picking detect with the panel's values (not
    while a render runs: it reads them per frame)."""
    from unleashed import core
    if not core.render_active:
        S.values['det_thresh'], S.values['det_size'] = thresh, size
        G.det_thresh, G.det_size = float(thresh), int(size)


def _clicked_index(value):
    """The index sent by a thumbnail's × (ui/theme.py: 'index:nonce')."""
    try:
        return int(str(value).split(':', 1)[0])
    except (TypeError, ValueError):
        return None


# ============================================================================ handlers: sources

def on_mode(mode):
    S.values['mode'] = mode
    n = len(G.INPUT_FACESETS)
    return [gr.Column(visible=S.MODES[mode] == 'selected'),
            gr.Button(visible=S.MODES[mode] == 'all_input' and n > 1), gr.Markdown(S.readiness()[1])]


def on_src_upload(files, tick, thresh, size, progress=gr.Progress()):
    if files:
        _apply_detection(thresh, size)
        _messages(S.add_sources([f.name if hasattr(f, 'name') else str(f) for f in files], progress))
    return [None] + _src_updates() + [_next_tick(tick)]


def on_src_path(path, tick, thresh, size):
    _apply_detection(thresh, size)
    _messages(S.add_source_path(path))
    return _src_updates() + [_next_tick(tick)]


def on_src_select(evt: gr.SelectData, tick):
    if evt is None or _is_echo('src', evt.index) or evt.index == S.active_source_index():
        return [gr.skip()] * 3
    S.select_source(evt.index)
    return [gr.Button(value=S.combine_label(), visible=len(S.same_person_photos()) >= 2),
            gr.Markdown(S.readiness()[1]), _next_tick(tick)]


def on_src_x(value, tick):
    i = _clicked_index(value)
    if i is None or not S.remove_source(i):
        return [gr.skip()] * 5
    return _src_updates() + [_next_tick(tick)]


def on_src_combine(tick):
    _info(S.combine_photo_sources())
    return _src_updates() + [_next_tick(tick)]


def on_src_shuffle(tick):
    S.shuffle_sources()
    return _src_updates() + [_next_tick(tick)]


# ============================================================================ handlers: targets

def on_tgt_upload(files, tick, progress=gr.Progress()):
    # the list's value: the files already listed plus the ones just dropped
    paths = [f.name if hasattr(f, 'name') else str(f) for f in (files or [])]
    _drop_missing_targets()
    known = set(S.target_paths())
    # the page's list may still show a file that is gone: not added again
    new = [p for p in paths if p not in known and os.path.isfile(p)]
    if new:
        _messages(S.add_targets(new, progress))
    return _tgt_updates() + [_next_tick(tick)]


def _clear_found_path(path):
    """Back to the start folder once the path was added; a wrong path stays to
    be fixed."""
    return S.path_start() if os.path.exists((path or '').strip()) else gr.skip()


def on_tgt_path(path, tick):
    _drop_missing_targets()
    _messages(S.add_target_path(path))
    return _tgt_updates() + [_next_tick(tick)]


def on_tgt_select(evt: gr.SelectData, tick):
    if evt is None:
        return [gr.skip()] * 10
    index = evt.index
    names = [t['name'] for t in S.targets]
    clicked = evt.value if isinstance(evt.value, str) else None
    if clicked and 0 <= index < len(names) and names[index] != clicked and names.count(clicked) == 1:
        index = names.index(clicked)            # a × was processed between the click and now
    if index == S.selected_target_index():
        return [gr.skip()] * 10
    S.select_target(index)
    return _tgt_updates()[1:] + [_next_tick(tick)]


def on_tgt_delete(files, tick):
    """A × in the list: the file the page took out goes."""
    remaining = {f.name if hasattr(f, 'name') else str(f) for f in (files or [])}
    # the list the page showed when the × was clicked: the newest list sent to
    # it that is one file more than what it has now, looking back only past
    # Adds. A file whose Add finished in the meantime is only in the newer
    # lists and stays (it was taken for a removed one when the newest list was
    # used). Otherwise (several × in one go): the newest list.
    lists = [set(l) for l in reversed(_sent_targets)]
    back = [lists[0]] if lists else []
    for older in lists[1:]:
        if not older <= back[-1]:
            break
        back.append(older)
    shown = next((l for l in back if remaining <= l and len(l - remaining) == 1),
                 next((l for l in lists if remaining <= l), set()))
    for path in [p for p in S.target_paths() if p in shown - remaining]:
        S.remove_target(path)
    # the whole list back: a second × while the first was pending was dropped
    # by the page, so page and server could disagree
    _drop_missing_targets()
    return _tgt_updates() + [_next_tick(tick)]


def on_tgt_clear(tick):
    S.clear_targets()
    return _tgt_updates() + [_next_tick(tick)]


def on_range(which, frame):
    if which == 'whole':
        S.reset_range()
    else:
        note = S.set_range(which, frame)
        if note:
            _info(note)
    return [gr.Markdown(S.range_text()), gr.Markdown(S.readiness()[1])]


def on_out_fps(value):
    S.set_out_fps(value)


# ============================================================================ handlers: people

_picker = []                # [(face, crop)] offered by the picker


def _specific_people():
    return next(k for k, v in S.MODES.items() if v == 'selected')


def _picked(face, crop, tick):
    """Add a face to the faces to replace; the mode follows (as it always did)."""
    problem = S.add_person(face, crop)
    if problem:
        gr.Warning(problem)
    _picker.clear()
    S.values['mode'] = _specific_people()
    return [gr.Dropdown(value=_specific_people()), gr.Column(visible=False), gr.Gallery(value=None)] + \
        _people_updates() + [gr.Markdown(S.readiness()[1]), _next_tick(tick)]


def on_use_face(frame, tick):
    found, problem = S.faces_in_frame(frame)
    if problem:
        gr.Warning(problem)
        return [gr.skip()] * 6
    if len(found) == 1:
        return _picked(*found[0], tick)
    _picker[:] = found
    return [gr.skip(), gr.Column(visible=True),
            gr.Gallery(value=[util.convert_to_gradio(crop) for _, crop in found], selected_index=None),
            gr.skip(), gr.skip(), gr.skip()]


def on_pick_face(evt: gr.SelectData, tick):
    if evt is None or not 0 <= evt.index < len(_picker):
        return [gr.skip()] * 6
    return _picked(*_picker[evt.index], tick)


def on_pick_cancel():
    _picker.clear()
    return [gr.Column(visible=False), gr.Gallery(value=None)]


def on_person_x(value, tick):
    i = _clicked_index(value)
    if i is None or not S.remove_person(i):
        return [gr.skip()] * 3
    return _people_updates() + [gr.Markdown(S.readiness()[1]), _next_tick(tick)]


# ============================================================================ handlers: painted mask

_painting = {'tid': None, 'frame': None}      # the file and frame the open editor paints on


def _current_frame(frame_num, tid=None):
    t = S.target(tid)
    if t is None or not os.path.isfile(t['path']):
        return None
    if t['kind'] == 'image':
        return get_image_frame(t['path'])
    return get_video_frame(t['path'], int(frame_num or 1))


def _editor_background(frame_num, tid=None):
    frame = _current_frame(frame_num, tid)
    return None if frame is None else util.convert_to_gradio_preview(frame)


def _paint_md_update():
    text = S.mask_info()
    return gr.Markdown(value=text, visible=bool(text))


def _paint_remove_update():
    return gr.Button(interactive=(S.target() or {}).get('mask') is not None)


def _paint_view(open_editor, editor_value=gr.skip()):
    """The editor in the preview's place; view, frame and range controls do
    nothing for it, so they hide until Done / Cancel."""
    t = S.target()
    video = t is not None and t['kind'] != 'image'
    return [gr.Column(visible=open_editor), gr.Image(visible=not open_editor), editor_value,
            _paint_md_update(), _paint_remove_update(),
            gr.Accordion(label=_title('Occlusion', 'occlusion')),
            gr.Row(visible=not open_editor), gr.Row(visible=video and not open_editor),
            gr.Row(visible=video and not open_editor)]


def on_paint_open(frame, tick):
    bg = _editor_background(frame)
    if bg is None:
        gr.Warning('Add a target file first')
        return _paint_view(False) + [tick]
    t = S.target()
    _painting.update(tid=t['id'], frame=int(frame or 1))
    layer = t['mask'] if t['mask'] is not None and t['mask'].shape[:2] == bg.shape[:2] else None
    return _paint_view(True, {'background': bg, 'layers': [layer] if layer is not None else [], 'composite': None}) + [tick]


def on_paint_done(editor, frame, tick):
    # on the file the editor was opened for, even if another one was clicked meanwhile
    tid = _painting['tid']
    t = S.target(tid) if tid is not None else None
    painted = t is not None and S.set_target_mask(editor, _painting['frame'], tid)
    _painting.update(tid=None, frame=None)
    if t is None:
        gr.Warning('The file was removed while painting: nothing saved')
    else:
        _info(f"Painted areas saved for {t['name']}" if painted else f"Nothing painted: no mask on {t['name']}")
    return _paint_view(False) + [_next_tick(tick)]


def on_paint_clear(frame, tick):
    bg = _editor_background(_painting['frame'] or frame, _painting['tid'])
    return _paint_view(True, {'background': bg, 'layers': [], 'composite': None}) + [tick]


def on_paint_cancel(tick):
    _painting.update(tid=None, frame=None)
    return _paint_view(False) + [_next_tick(tick)]


def on_paint_remove(tick):
    S.clear_target_mask()
    _painting.update(tid=None, frame=None)
    return _paint_view(False) + [_next_tick(tick)]


# ============================================================================ preview

def _image(value, note):
    """The preview image; its label only when there is something to say (no
    file name: the target list marks the file shown)."""
    return gr.Image(value=value, label=note or 'Preview', show_label=bool(note), visible=True)


# every section box and the summary in its title
HEADERS = [('acc_src', 'Source faces', 'sources'), ('acc_tgt', 'Target files', 'targets'),
           ('acc_faces', 'Faces to replace', 'faces'), ('acc_swap', 'Swap', 'swap'),
           ('acc_expr', 'Expression', 'expression'), ('acc_occ', 'Occlusion', 'occlusion'),
           ('acc_enh', 'Enhance', 'enhance'), ('acc_det', 'Detection & tracking', 'detection'),
           ('acc_vid', 'Video output', 'video')]
HEADER_KEYS = [k for k, _, _ in HEADERS]


def _headers():
    return [gr.Accordion(label=_title(name, key)) for _, name, key in HEADERS]


def on_preview(data, force=False):
    from unleashed import core
    vals = _vals(data)
    view, frame_num, auto = data[C['view']], int(data[C['frame']] or 1), data[C['auto']]

    def work():
        S.apply_settings(vals)
        if _painting['tid'] is not None:
            return gr.skip()               # the editor stands in for the preview: nothing to show or compare
        return _render_view(view, frame_num, auto or force)

    # "tick|state|nonce|change" for the browser's preview badge (ui/theme.py):
    # the tick this answer belongs to, what happened to the picture and how it
    # differs from the picture shown before (see _compare_with_last)
    def done(state):
        return f"{data[C['tick']]}|{state}|{time.time_ns()}|{_rendered['change']}"

    _rendered.update(kind='empty', change='')
    try:
        ran, image = core.preview_locked(work)
    except Exception as e:
        traceback.print_exc()
        return ([gr.Image(label=f'Preview failed: {e}', show_label=True), gr.skip()] + [gr.skip()] * len(HEADERS)
                + [done('failed')])
    if not ran:
        return ([gr.Image(label='Preview paused while rendering', show_label=True), gr.skip()] + [gr.skip()] * len(HEADERS)
                + [done('paused')])
    if _painting['tid'] is not None:
        image = gr.skip()                            # the editor stands in for the preview until Done / Cancel
    state = 'same' if image == gr.skip() else _rendered['kind']
    return [image, gr.Markdown(S.readiness()[1])] + _headers() + [done(state)]


# what the last preview showed, for the badge's "n % changed" (previews run one
# at a time: concurrency_id fs_preview)
_rendered = {'kind': 'empty', 'change': ''}
_last_picture = {'context': None, 'image': None}


def _compare_with_last(context, image):
    """'' (nothing to compare: another file / frame / view), 'identical',
    'tiny' (only differences of 1-2 levels) or the share of pixels that changed
    by more than 2 levels, against the previous picture of the same context."""
    last_context, last = _last_picture['context'], _last_picture['image']
    _last_picture.update(context=context, image=image)
    if context != last_context or last is None or last.shape != image.shape:
        return ''
    import cv2
    diff = cv2.absdiff(image, last)
    if diff.ndim == 3:
        diff = cv2.max(cv2.max(diff[:, :, 0], diff[:, :, 1]), diff[:, :, 2])
    if cv2.countNonZero(diff) == 0:
        return 'identical'
    changed = cv2.countNonZero(cv2.compare(diff, 2, cv2.CMP_GT)) / float(diff.size)
    return f'{changed:.6f}' if changed > 0 else 'tiny'


# The last preview swap, so Swapped <-> Side by side (and Refresh with
# nothing changed) show it again instead of swapping the frame again. Reused
# only when everything the swap reads is the same: the frame's pixels, the
# preview options (painting included), every simple engine / app setting,
# and the source and picked faces (embeddings, mask offsets).
_swap_cache = {'key': None, 'image': None}


def _swap_key(frame, options):
    import hashlib
    h = hashlib.md5()

    def add(x, depth=0):
        if isinstance(x, np.ndarray):
            h.update(repr((x.shape, x.dtype.str)).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict) and depth < 4:
            for k in sorted(x, key=str):
                h.update(repr(k).encode())
                add(x[k], depth + 1)
        elif isinstance(x, (list, tuple)) and depth < 4:
            h.update(b'[')
            for v in x:
                add(v, depth + 1)
            h.update(b']')
        else:
            h.update(repr(x).encode())

    simple = (bool, int, float, str, type(None), list, tuple)

    def settings_of(obj, skip=()):
        return {k: v for k, v in vars(obj).items()
                if not k.startswith('_') and k not in skip and isinstance(v, simple)}

    add(frame)
    add(vars(options))
    add(settings_of(G, skip=('INPUT_FACESETS', 'TARGET_FACES')))
    add(settings_of(G.CFG))
    for fs in G.INPUT_FACESETS:
        h.update(repr(id(fs)).encode())
        for f in getattr(fs, 'faces', []):
            add(getattr(f, 'embedding', None))
            add(getattr(f, 'mask_offsets', None))
    for f in G.TARGET_FACES:
        add(getattr(f, 'embedding', None))
    return h.hexdigest()


def _render_view(view, frame_num, swap_now):
    """The preview image update for this view."""
    from unleashed import core
    t = S.target()
    if t is None:
        return _image(None, 'add a target file')
    if not os.path.isfile(t['path']):
        return _image(None, 'this file is gone (temp folder cleaned?): add it again')
    t_start = time.perf_counter()
    frame = get_image_frame(t['path']) if t['kind'] == 'image' else get_video_frame(t['path'], frame_num)
    if frame is None:
        return _image(None, 'this frame could not be read')

    note = ''
    shown = frame
    reused = False
    t_swap = time.perf_counter()
    if view == 'Mask':
        shown = core.mask_view(frame, S.build_options(mask_view=True))
        plugin, warn = S.mask_plugin()
        note = 'green = swapped' + (f' ({warn})' if warn else '' if plugin else ' (no occlusion mask)')
    elif view in ('Swapped', 'Side by side'):
        if not swap_now:
            return gr.skip()                              # auto-update off: Refresh does it
        if not G.INPUT_FACESETS:
            note = 'add a source face'
        elif S.MODES[S.values['mode']] == 'selected' and not G.TARGET_FACES:
            note = 'no faces picked yet'
        else:
            options = S.build_options()
            key = _swap_key(frame, options)
            reused = key == _swap_cache['key']
            if reused:
                swapped = _swap_cache['image']
            else:
                swapped = core.live_swap(frame.copy(), options)
                _swap_cache.update(key=key, image=swapped)
            if swapped is None or np.array_equal(swapped, frame):
                note = f"no face swapped ({S.values['mode']})"
            if view == 'Side by side':
                shown = np.hstack([frame, swapped if swapped is not None else frame])
                note = note or 'original | swapped'
            else:
                shown = swapped if swapped is not None else frame
    image = util.convert_to_gradio_preview(shown)
    _rendered.update(kind='shown', change=_compare_with_last((t['path'], frame_num if t['kind'] != 'image' else 0, view),
                                                              image))
    if view != 'Original':
        # server-side time; if the browser shows it much later, the rest is the tunnel / network
        t_done = time.perf_counter()
        print(f'[preview] frame {frame_num}: load {(t_swap - t_start) * 1000:.0f} ms | {view.lower()} '
              f'{(t_done - t_swap) * 1000:.0f} ms{" (swap reused)" if reused else ""} | '
              f'total {(t_done - t_start) * 1000:.0f} ms', flush=True)
    return _image(image, note)


# ============================================================================ run

_pending = {}
_disconnect_after = False   # this run: "Disconnect Colab when done"
_starting = 0.0             # when a Start was accepted (0: none); the render clears it


def _rendering():
    # the render takes run_lock right after the check; a Start whose render
    # never came (page closed in between) stops counting after 30 s
    return S.run_lock.locked() or (_starting > 0 and time.time() - _starting < 30)



def on_start_check(data):
    """Refuse clearly before anything happens; otherwise hand the panel values
    to the render. A refused Start leaves a running render and its buttons alone."""
    if _rendering():
        raise gr.Error('A render is already running')
    if _painting['tid'] is not None:
        raise gr.Error('Press Done or Cancel on the painting first')
    S.values.update(_vals(data))
    ready, text = S.readiness()
    if not ready:
        raise gr.Error(text.replace('To start: ', 'Cannot start yet: '))
    missing = [t['name'] for t in S.targets if not os.path.isfile(t['path'])]
    if missing:
        raise gr.Error('These files are gone (moved, deleted, or the temp folder was cleaned): '
                       + ', '.join(missing[:5]) + '. Remove them from Target files, or add them again.')
    if any(t['kind'] != 'image' for t in S.targets):
        if not util.is_installed('ffmpeg'):
            raise gr.Error('ffmpeg is not installed: videos cannot be rendered')
        # pairs ffmpeg refuses only once the render is under way
        codec, fmt = G.CFG.output_video_codec, G.CFG.output_video_format
        if fmt == 'webm' and codec != 'libvpx-vp9':
            raise gr.Error(f'WebM videos need the libvpx-vp9 codec (Video Codec is {codec}): change one of them in Settings')
        if codec.endswith('_nvenc') and not any('CUDA' in p for p in (G.execution_providers or [])):
            raise gr.Error(f'{codec} needs an NVIDIA GPU: choose libx264 (or libx265) as Video Codec in Settings')
    global _starting, _disconnect_after
    from unleashed import core
    colab.cancel()                  # a new run: whatever the last one scheduled is off
    _disconnect_after = bool(data.get(C['disconnect'])) and colab.available()
    _pending.clear()
    _pending.update(_vals(data))
    core.stop_requested = False
    _starting = time.time()
    return [gr.Button(interactive=False), gr.Button(interactive=True), gr.Markdown('Rendering …')]


def on_render(progress=gr.Progress()):
    global _starting
    from unleashed import core
    from ui.main import prepare_environment
    if not S.run_lock.acquire(blocking=False):
        return gr.Markdown('A render is already running')
    started = time.time()
    try:
        prepare_environment()
        out_dir = G.output_path                   # where this run writes, even if Settings change meanwhile
        entries = S.process_entries()
        with core._swap_lock:                     # waits for a preview; none can start after this
            S.apply_settings(_pending)
            core.render_active = True
            # everything the render uses, read here: the panel's handlers
            # (Factory defaults, source edits) may change S.values from now on
            v = dict(S.values)
            plugin, warn = S.mask_plugin()
            source = S.active_source_index()
        try:
            if G.CFG.clear_output:
                S.clear_output_media()
            G.video_encoder = G.CFG.output_video_codec
            G.video_quality = G.CFG.video_quality
            G.max_memory = G.CFG.memory_limit if G.CFG.memory_limit > 0 else None
            if warn:
                gr.Warning(warn)
            core.batch_process_regular(S.SWAP_MODEL, "File", entries, plugin, v['mask_objects'],
                                       v['method'] == S.METHOD_MEMORY, None, bool(v['keep_mouth']),
                                       bool(v['keep_eyes']), int(v['passes']), progress, source)
        finally:
            core.render_active = False
        finished = [e.finalname for e in entries if getattr(e, 'completed', False) and os.path.isfile(e.finalname)]
        S.record_outputs(finished)
        secs = time.time() - started
        done = f'{len(finished)} of {len(entries)} file{"s" if len(entries) != 1 else ""}'
        stopped = not G.processing
        G.processing = False
        if stopped and len(finished) < len(entries):
            partial = [os.path.basename(e.finalname) for e in entries
                       if not getattr(e, 'completed', False) and e.finalname and os.path.isfile(e.finalname)]
            text = (f'**Stopped** after {secs:.0f} s: {done} finished, saved in `{out_dir}`.' if finished
                    else f'**Stopped** after {secs:.0f} s, before a file was finished.')
            if partial:
                text += f' The interrupted part is left as `{partial[0]}` (no sound; the next run overwrites it).'
            return gr.Markdown(text + _disconnect_note(entries, finished, stopped, out_dir))
        what = f'`{os.path.basename(finished[0])}`' if len(finished) == 1 else done
        text = f'**Done** in {secs:.0f} s: {what} saved in `{out_dir}`.'
        silent = [os.path.basename(e.finalname) for e in entries if getattr(e, 'completed', False) and getattr(e, 'no_audio', False)]
        if silent:
            text += f' Saved without sound (it could not be copied): {", ".join(silent[:3])}.'
        failed = [f'{os.path.basename(e.filename)}: {e.failure}' for e in entries
                  if not getattr(e, 'completed', False) and getattr(e, 'failure', '')]
        if failed:
            text = text.replace('**Done**', '**Done, with problems**') + ' Not saved: ' + '; '.join(failed[:3]) + '.'
        return gr.Markdown(text + _disconnect_note(entries, finished, stopped, out_dir))
    except Exception as e:
        traceback.print_exc()
        G.processing = False
        # the first line (ffmpeg's errors run on for a page): the rest is in the console
        text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        if 'FFMPEG encountered' in str(e):
            text = 'ffmpeg stopped while writing the video (codec / format / disk space?)'
        return gr.Markdown(f'**Render failed:** {text[:300]} (details in the console)'
                           + (' Colab stays connected.' if _disconnect_after else ''))
    finally:
        core.stop_requested = False
        _starting = 0.0
        S.run_lock.release()


def _disconnect_note(entries, finished, stopped, out_dir):
    """ " Colab disconnects in 60 s ..." when the run asked for it and every
    file is safely saved on Drive (unleashed/colab.py), else why it stays."""
    if not _disconnect_after:
        return ''
    ok, why = colab.check(entries, finished, stopped, out_dir)
    if ok and colab.schedule():
        return (f' **Colab disconnects in {colab.DELAY} s**, after Google Drive has the files'
                ' (*Stay connected* to keep it).')
    return f' Colab stays connected: {why}.'


def on_render_done(tick):
    running = _rendering()
    return [gr.Button(interactive=not running), gr.Button(interactive=running),
            gr.Markdown(S.readiness()[1]), _next_tick(tick), gr.Button(visible=colab.pending())]


def on_stay_connected():
    colab.cancel()
    return [gr.Button(visible=False), gr.Markdown('Colab stays connected.')]


def on_stop():
    from unleashed import core
    if not _rendering():
        return [gr.Button(interactive=False), gr.Markdown('Nothing is rendering.')]
    core.stop_requested = True        # also when the render is still starting
    G.processing = False
    return [gr.Button(interactive=False), gr.Markdown('Stopping: the frames being processed finish first ...')]


# ============================================================================ defaults

def on_save_defaults(data):
    S.values.update(_vals(data))
    try:
        path = S.save_defaults()
    except OSError as e:
        raise gr.Error(f'Could not save: {e}')
    _info('Saved as your defaults (loaded at every start)')
    return gr.Button(interactive=True)


def on_load_defaults(keys, tick):
    if not S.load_my_defaults():
        gr.Warning('No saved defaults yet: set the panel up and press "Save my defaults"')
        return [gr.skip()] * (len(keys) + 1)
    _info('Your saved defaults are back')
    return [S.values[k] for k in keys] + [_next_tick(tick)]


def on_factory_defaults(keys, tick):
    S.factory_defaults()
    _info('Factory defaults loaded (your saved defaults are kept)' if S.has_saved_defaults() else 'Factory defaults loaded')
    return [S.values[k] for k in keys] + [_next_tick(tick)]


# ============================================================================ used by other tabs

def add_faceset_to_sources(path):
    """Face Management 'Use in Face Swap'. Returns refresh_values() for
    refresh_outputs()."""
    if path and os.path.isfile(path):
        _messages(S.add_sources([path]))
    else:
        gr.Warning('Save the faceset first')
    return refresh_values()
