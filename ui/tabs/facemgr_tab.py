"""Face Management: build a faceset (.fsz) of one person.

Flow: 1 add photos (or faces from a video frame, or open a faceset to edit)
-> 2 review: every photo is checked as it comes in; its status is on the photo
itself, and clicking a photo shows why, with its own Remove / Keep buttons
-> 3 save under a name.

Photos are identified by the photo itself, never by a number, so nothing
renumbers when one is removed. Every removal can be undone.
"""
import hashlib
import os
import re
import shutil
import tempfile
import cv2
import numpy as np
import gradio as gr
import unleashed.utilities as util
import unleashed.globals
from unleashed import faceset_check
from unleashed.face_util import extract_face_images, get_all_faces
from unleashed.capturer import get_video_frame, get_video_frame_total

# The photos in the faceset, in order. Each: {'id': int, 'image': the 512 BGR
# cut-out saved in the .fsz, 'face': the face found in it, 'thumb': path of a
# small JPEG for the gallery, 'metrics': faceset_check.photo_metrics,
# 'source': where it came from, 'hash': of the source file, 'kept': warning
# ignored by the user}.
entries = []
_rows = {}              # id -> faceset_check row for the current list
_view = []              # ids shown in the gallery (the filter), in order
_selected = None        # id of the photo shown on the right
_undo = []              # [(label, [(position, entry), ...]) or (label, ('added', [ids]))], newest last
_next_id = 0
_video_path = None
current_video_fps = 0
_last_saved = None          # the .fsz written by the last Save (for 'Use in Face Swap')
_saved_ids = None           # the photos (ids, in order) that Save wrote

SHOW_ALL, SHOW_ATTENTION = 'All photos', 'Needs attention'
_filter = SHOW_ALL

FLAG_TEXT = {           # flag prefix -> (label on the photo, explanation)
    'other person?': ('Other person?',
                      "This face does not match the rest of the faceset. If it is someone else, remove it: "
                      "a wrong face pulls the averaged identity away."),
    'unlike the others': ('Looks different',
                          "Probably the same person, but unlike most of the other photos (another age, makeup, "
                          "lighting or filter?). Keep it only if that look is what you want."),
    'duplicate': ('Duplicate',
                  "Nearly identical to another photo (shown below). A repeated shot gives that one moment extra "
                  "weight; one of them is enough. The better of the two stays unmarked."),
    'small': ('Too small',
              "The face was small in the original photo. The identity model looks at 112 px, so a small face "
              "carries little detail (128 px or more is better)."),
    'blurry': ('Blurry', "Much less sharp than the other photos."),
    'no face data': ('No face data', "No identity could be read from this face."),
}


def facemgr_tab():
    """Builds the tab; returns register_load(ui) for the page-load refresh."""
    with gr.Tab("Face Management"):
        gr.Markdown("## Build a faceset\nA faceset (.fsz) holds photos of **one person**. The swap blends them into one "
                    "identity, so clean photos matter more than many: aim for 30–80 good photos of one look "
                    "(same age, hair, makeup), mostly frontal plus some turned to either side.")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=280):
                gr.Markdown("### 1 · Add photos")
                in_photos = gr.Files(label="Drop photos of the person (JPG / PNG) or facesets (.fsz)",
                                     file_count="multiple", file_types=["image", ".fsz"], height=150)
                with gr.Accordion("Faces from a video", open=False):
                    in_video = gr.File(label="Video", file_types=["video"], height=90)
                    video_frame = gr.Image(label="Frame", interactive=False, format="jpeg", height=220)
                    video_slider = gr.Slider(1, 1, value=1, step=1, label="Frame", interactive=False)
                    btn_add_frame = gr.Button("Add the faces in this frame", interactive=False)
                with gr.Accordion("Add a faceset", open=False):
                    in_fsz = gr.File(label="A faceset (.fsz): its photos are added to the list",
                                     file_types=[".fsz"], height=90)
                gr.Markdown("### 3 · Save")
                save_name = gr.Textbox(value="faceset", label="Faceset name", max_lines=1)
                btn_save = gr.Button("Save faceset", variant="primary", interactive=False)
                save_file = gr.File(label="Saved faceset", interactive=False, visible=False)
                save_msg = gr.Markdown()
                btn_use_in_swap = gr.Button("Use in Face Swap", visible=False)
            with gr.Column(scale=6, min_width=420):
                gr.Markdown("### 2 · Review")
                summary = gr.Markdown(_summary_text())
                with gr.Row(equal_height=True):
                    show = gr.Radio([SHOW_ALL, SHOW_ATTENTION], value=SHOW_ALL, show_label=False, container=False,
                                    scale=3, min_width=260)
                    btn_remove_flagged = gr.Button("Remove flagged photos", variant="stop", size="sm", interactive=False,
                                                   scale=2, min_width=160)
                    btn_undo = gr.Button("Undo", size="sm", interactive=False, scale=2, min_width=120)
                # no fixed height: with one, Gradio 5.9.1 clips the grid instead of
                # scrolling it; the auto-height gallery scrolls (height set in ui/theme.py)
                gallery = gr.Gallery(show_label=False, columns=5, object_fit="cover",
                                     allow_preview=False, preview=False, interactive=False,
                                     elem_id="facemgr_gallery")
                with gr.Row():
                    btn_start_over = gr.Button("Start over", size="sm")
                    btn_confirm_clear = gr.Button("Yes, remove all photos", variant="stop", size="sm", visible=False)
                    btn_cancel_clear = gr.Button("Cancel", size="sm", visible=False)
            with gr.Column(scale=3, min_width=280):
                gr.Markdown("### Selected photo")
                sel_image = gr.Image(show_label=False, interactive=False, height=260, format="jpeg")
                sel_info = gr.Markdown("Click a photo to see its details.")
                with gr.Row():
                    btn_remove = gr.Button("Remove this photo", variant="stop", interactive=False)
                    btn_keep = gr.Button("Keep anyway", interactive=False)
                dup_image = gr.Image(label="Nearly identical to", interactive=False, height=150, format="jpeg",
                                     visible=False)

    view = [gallery, summary, btn_remove_flagged, btn_undo, btn_save, sel_image, sel_info, btn_remove, btn_keep, dup_image,
            btn_use_in_swap]
    detail = [sel_image, sel_info, btn_remove, btn_keep, dup_image]
    one = dict(concurrency_id='facemgr', concurrency_limit=1)

    in_photos.upload(fn=on_photos_added, inputs=[in_photos], outputs=[in_photos] + view, **one)
    in_video.upload(fn=on_video_loaded, inputs=[in_video], outputs=[video_frame, video_slider, btn_add_frame], **one)
    in_video.clear(fn=on_video_cleared, outputs=[video_frame, video_slider, btn_add_frame], **one)
    video_slider.release(fn=on_video_frame, inputs=[video_slider], outputs=[video_frame], **one)
    btn_add_frame.click(fn=on_add_frame, inputs=[video_slider], outputs=view, **one)
    in_fsz.upload(fn=on_faceset_opened, inputs=[in_fsz], outputs=[in_fsz, save_name] + view, **one)
    gallery.select(fn=on_photo_selected, outputs=detail, **one)
    show.change(fn=on_show_changed, inputs=[show], outputs=view, **one)
    btn_remove.click(fn=on_remove_selected, outputs=view, **one)
    btn_keep.click(fn=on_keep_selected, outputs=view, **one)
    btn_remove_flagged.click(fn=on_remove_flagged, outputs=view, **one)
    btn_undo.click(fn=on_undo, outputs=view, **one)
    btn_start_over.click(fn=lambda: [gr.Button(visible=False), gr.Button(visible=True), gr.Button(visible=True)],
                         outputs=[btn_start_over, btn_confirm_clear, btn_cancel_clear], **one)
    btn_cancel_clear.click(fn=lambda: [gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False)],
                           outputs=[btn_start_over, btn_confirm_clear, btn_cancel_clear], **one)
    btn_confirm_clear.click(fn=on_start_over, outputs=[btn_start_over, btn_confirm_clear, btn_cancel_clear, save_file, save_msg] + view, **one)
    btn_save.click(fn=on_save, inputs=[save_name], outputs=[save_file, save_msg, btn_use_in_swap], **one)
    from ui.tabs import faceswap_tab as fs
    # in the Face Swap tab's group: it changes that tab's source list
    # only the list as it was saved: after a change, Save first
    btn_use_in_swap.click(fn=lambda: fs.add_faceset_to_sources(_last_saved if _saved_is_current() else None),
                          outputs=fs.refresh_outputs(),
                          concurrency_id='fs_state', concurrency_limit=1).then(
        fn=fs.src_highlight,
        outputs=fs.C['src_gal'], show_progress='hidden')

    def register_load(ui):
        """A reloaded page (or Restart Server) shows the server's list: the tab
        had no load event, so it came back empty while the server kept the
        photos, with Save disabled and the filter out of step."""
        def on_page_load():
            _refresh()
            saved = _last_saved if _last_saved and os.path.isfile(_last_saved) else None
            return _render() + [gr.Radio(value=_filter), gr.File(value=saved, visible=bool(saved)), gr.Markdown('')]
        ui.load(on_page_load, None, view + [show, save_file, save_msg], show_progress='hidden', **one)

    return register_load


# ----------------------------------------------------------------------------- state

_thumbs = None


def _thumb_dir():
    """Folder for the gallery thumbnails; emptied once per app start."""
    global _thumbs
    if _thumbs is None:
        _thumbs = os.path.join(os.environ.get('TEMP') or tempfile.gettempdir(), 'facemgr_thumbs')
        shutil.rmtree(_thumbs, ignore_errors=True)
        os.makedirs(_thumbs, exist_ok=True)
    return _thumbs


def _write_thumb(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])


def _add(face, image, source, file_hash=None):
    global _next_id
    _next_id += 1
    thumb = os.path.join(_thumb_dir(), f'{_next_id}.jpg')
    _write_thumb(thumb, image)
    entries.append({'id': _next_id, 'image': image, 'face': face, 'thumb': thumb, 'source': source,
                    'hash': file_hash, 'kept': False,
                    'metrics': faceset_check.photo_metrics(face, image)})


def _by_id(i):
    return next((e for e in entries if e['id'] == i), None)


def _needs_attention(e):
    r = _rows.get(e['id'])
    return bool(r and r['removable'] and not e['kept'])


def _refresh():
    """Check the whole list again and rebuild the gallery's view."""
    global _rows, _view, _selected
    rows = faceset_check.check([e['metrics'] for e in entries]) if entries else []
    _rows = {e['id']: r for e, r in zip(entries, rows)}
    for e, r in zip(entries, rows):
        r['dup_id'] = entries[r['dup_of']]['id'] if r.get('dup_of') is not None else None
    _view = [e['id'] for e in entries if _filter == SHOW_ALL or _needs_attention(e)]
    if _selected is not None and _selected not in _view:
        _selected = None


def _flag_key(flag):
    return next((k for k in FLAG_TEXT if flag.startswith(k)), None)


def _caption(e):
    r = _rows.get(e['id'])
    if not r:
        return ''
    keys = [k for k in (_flag_key(f) for f in r['flags'] if f.startswith(faceset_check.REMOVABLE)) if k]
    if not keys:
        return '✓'
    if e['kept']:
        return '✓ kept'
    keys.sort(key=list(FLAG_TEXT).index)          # the most important reason first
    return '⚠ ' + FLAG_TEXT[keys[0]][0] + (f' +{len(keys) - 1}' if len(keys) > 1 else '')


def _summary_text():
    n = len(entries)
    if n == 0:
        return ("**No photos yet.** Add photos of the person on the left. Each photo is checked as it comes in; "
                "problems show on the photo itself.")
    attention = [e for e in entries if _needs_attention(e)]
    counts = {}
    for e in attention:
        for f in _rows[e['id']]['flags']:
            k = _flag_key(f)
            if k and f.startswith(faceset_check.REMOVABLE):
                counts[FLAG_TEXT[k][0].lower()] = counts.get(FLAG_TEXT[k][0].lower(), 0) + 1
    text = f"**{n} photo{'s' if n != 1 else ''}** · ✓ {n - len(attention)} fine"
    if attention:
        text += f" · ⚠ {len(attention)} need attention (" + ', '.join(f'{c} {k}' for k, c in counts.items()) + ")"
    rows = [_rows[e['id']] for e in entries]
    used = sum(1 for r in rows if r['shape'] in ('yes', 'fill'))
    if any(r['shape'] is not None for r in rows):
        text += f" · face shape uses {used}"
    sims = [r['sim'] for r in rows if r['sim'] is not None]
    if len(sims) >= 3 and np.median(sims) < 0.30:
        text += "\n\n⚠ **These photos do not look like one person.** A faceset should hold one person only."
    elif attention and _filter == SHOW_ALL:
        text += "\n\nClick a ⚠ photo to see why, or choose *Needs attention* to see only those."
    elif not attention and _filter == SHOW_ATTENTION:
        text += "\n\nNothing needs attention. Choose *All photos* to see every photo."
    return text


def _detail():
    """Right-hand panel for the selected photo."""
    e = _by_id(_selected) if _selected is not None else None
    if e is None:
        return [gr.Image(value=None), gr.Markdown("Click a photo to see its details."),
                gr.Button(interactive=False), gr.Button(value="Keep anyway", interactive=False),
                gr.Image(value=None, visible=False)]
    r, m = _rows.get(e['id'], {}), e['metrics']
    lines = []
    keys = sorted({k for k in (_flag_key(f) for f in r.get('flags', [])) if k}, key=list(FLAG_TEXT).index)
    if keys and e['kept']:
        lines.append(f"**✓ Kept** despite: {', '.join(FLAG_TEXT[k][0].lower() for k in keys)}.")
    elif keys:
        for k in keys:
            extra = ''
            if k == 'small' and m.get('size'):
                extra = f" Here: {m['size']:.0f} px."
            if k == 'blurry' and r.get('sharp_rel') is not None:
                extra = f" Here: {100 * r['sharp_rel']:.0f}% of this faceset's typical sharpness."
            label = FLAG_TEXT[k][0]
            lines.append(f"**⚠ {label}{'' if label.endswith('?') else '.'}** {FLAG_TEXT[k][1]}{extra}")
    else:
        lines.append("**✓ Looks fine.**")
    facts = []
    if r.get('sim') is not None:
        typical = f" (typical here: {r['median_sim']:.2f})" if r.get('median_sim') is not None else ''
        facts.append(f"Match with the other photos: {r['sim']:.2f}{typical}")
    if m.get('yaw') is not None:
        facts.append(f"Head: turned {m['yaw']:.0f}°, up/down {m['pitch']:.0f}° · mouth "
                     f"{'open' if m['mouth'] > _mouth_open() else 'closed'}")
    if m.get('size'):
        facts.append(f"Face size in the original photo: {m['size']:.0f} px")
    shape = r.get('shape')
    if shape == 'yes':
        facts.append("Face shape from source: used")
    elif shape == 'fill':
        facts.append("Face shape from source: used to reach 3 photos (no better ones)")
    elif shape == 'no':
        facts.append("Face shape from source: not used (" + _shape_reason(m, r) + ")")
    facts.append(f"From: {e['source']}")
    text = '\n\n'.join(lines) + '\n\n' + '\n'.join(f'- {f}' for f in facts)
    dup = _by_id(r.get('dup_id')) if r.get('dup_id') is not None else None
    return [gr.Image(value=cv2.cvtColor(e['image'], cv2.COLOR_BGR2RGB)), gr.Markdown(text),
            gr.Button(interactive=True),
            gr.Button(value="Undo keep" if e['kept'] else "Keep anyway", interactive=bool(keys)),
            gr.Image(value=cv2.cvtColor(dup['image'], cv2.COLOR_BGR2RGB), visible=True) if dup else gr.Image(value=None, visible=False)]


def _shape_reason(m, r):
    from unleashed.face_shape import MOUTH_OPEN, SHAPE_PITCH, SHAPE_YAW
    reasons = []
    if any(f.startswith('other person?') for f in r.get('flags', [])):
        reasons.append('another person')
    if m.get('yaw') is not None:
        if m['yaw'] > SHAPE_YAW:
            reasons.append(f"turned more than {SHAPE_YAW:.0f}°")
        if m['pitch'] > SHAPE_PITCH:
            reasons.append(f"up/down more than {SHAPE_PITCH:.0f}°")
        if m['mouth'] > MOUTH_OPEN:
            reasons.append('mouth open')
    return ', '.join(reasons) or 'enough better photos'


def _mouth_open():
    from unleashed.face_shape import MOUTH_OPEN
    return MOUTH_OPEN


def _render():
    """Values for the `view` outputs: gallery, summary, buttons, detail panel."""
    shown = [e for i in _view for e in [_by_id(i)] if e is not None]
    for e in shown:
        if not os.path.isfile(e['thumb']):          # the temp folder was cleaned
            _write_thumb(e['thumb'], e['image'])
    selected_index = _view.index(_selected) if _selected in _view else None
    flagged = sum(1 for e in entries if _needs_attention(e))
    undo_label = f"Undo: {_undo[-1][0]}" if _undo else "Undo"
    return [gr.Gallery(value=[(e['thumb'], _caption(e)) for e in shown], selected_index=selected_index),
            gr.Markdown(_summary_text()),
            gr.Button(value=f"Remove {flagged} flagged photo{'s' if flagged != 1 else ''}" if flagged else "Remove flagged photos",
                      interactive=flagged > 0),
            gr.Button(value=undo_label, interactive=bool(_undo)),
            gr.Button(interactive=bool(entries)),
            *_detail(),
            gr.Button(visible=_saved_is_current())]


def _saved_is_current():
    """The last saved .fsz still holds exactly the photos in the list."""
    return (bool(_last_saved) and os.path.isfile(_last_saved) and _saved_ids is not None
            and _saved_ids == [e['id'] for e in entries])


def _remove(ids, label):
    """Take the photos out, remembering where they were (for Undo)."""
    removed = [(k, e) for k, e in enumerate(entries) if e['id'] in ids]
    if not removed:
        return
    entries[:] = [e for e in entries if e['id'] not in ids]
    _undo.append((label, removed))
    del _undo[:-20]


def _select_after_removal(position):
    """Select the photo that took the removed one's place in the gallery."""
    global _selected
    _selected = _view[min(position, len(_view) - 1)] if _view else None


# ----------------------------------------------------------------------------- events

def on_photos_added(files, progress=gr.Progress()):
    """Add the faces of dropped photos (and the photos of dropped .fsz
    facesets); the drop zone is emptied again."""
    if not files:
        return [None] + _render()
    paths = [f.name if hasattr(f, 'name') else str(f) for f in files]
    for path in [p for p in paths if p.lower().endswith('.fsz')]:
        _add_faceset_file(path, progress)
    files = [p for p in paths if not p.lower().endswith('.fsz')]
    known = {e['hash'] for e in entries if e['hash']}
    skipped, no_face, added = [], [], 0
    for k, path in enumerate(files):
        progress(k / len(files), desc="Adding photos")
        with open(path, 'rb') as fh:
            digest = hashlib.sha1(fh.read()).hexdigest()
        if digest in known:
            skipped.append(os.path.basename(path))
            continue
        known.add(digest)
        found = extract_face_images(path, (False, 0), 0.5)
        if not found:
            no_face.append(os.path.basename(path))
        for face, image in found:
            _add(face, image, os.path.basename(path), digest)
            added += 1
    _refresh()
    if no_face:
        gr.Warning(f"No face found in {len(no_face)} photo(s): {', '.join(no_face[:5])}")
    if skipped:
        gr.Info(f"Already in the list, not added again: {', '.join(skipped[:5])}")
    return [None] + _render()


def on_video_loaded(video):
    global _video_path, current_video_fps
    if video is None:
        return on_video_cleared()
    _video_path = video.name if hasattr(video, 'name') else str(video)
    total = max(1, int(get_video_frame_total(_video_path) or 1))
    current_video_fps = util.detect_fps(_video_path) or 1
    frame = get_video_frame(_video_path, 1, exact=True)
    return [gr.Image(value=None if frame is None else util.convert_to_gradio(frame)),
            gr.Slider(value=1, minimum=1, maximum=total, step=1, interactive=True),
            gr.Button(interactive=frame is not None)]


def on_video_cleared():
    global _video_path
    _video_path = None
    # step=1: without it Gradio 5.9.1 fails on minimum == maximum ('math domain error')
    return [gr.Image(value=None), gr.Slider(value=1, minimum=1, maximum=1, step=1, interactive=False), gr.Button(interactive=False)]


def on_video_frame(frame_num):
    if _video_path is None:
        return gr.Image()
    frame = get_video_frame(_video_path, int(frame_num), exact=True)
    return gr.Image(value=None if frame is None else util.convert_to_gradio(frame))


def on_add_frame(frame_num):
    if _video_path is None:
        return _render()
    found = extract_face_images(_video_path, (True, int(frame_num)), 0.5)
    if not found:
        gr.Warning('No face found in this frame')
    for face, image in found:
        _add(face, image, f"{os.path.basename(_video_path)}, frame {int(frame_num)}")
    _refresh()
    return _render()


def _add_faceset_file(path, progress):
    """Add the photos of a .fsz to the list (the list is kept: add several
    facesets, photos and video frames, then sort them out). Photos already in
    the list are not added again. Undo takes the added photos out. Returns
    the number of photos added (None: the file could not be read)."""
    name = os.path.basename(path)
    folder = tempfile.mkdtemp(prefix='faceset_')
    missing, skipped, added = [], 0, []
    try:
        try:
            util.unzip(path, folder)
        except Exception as e:
            gr.Warning(f"{name} is damaged or not a faceset ({e}); nothing was added")
            return None
        known = {e['hash'] for e in entries if e['hash']}
        pngs = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
        pngs.sort(key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', f)])
        for k, file in enumerate(pngs):
            progress(k / max(len(pngs), 1), desc=f"Adding {name}")
            p = os.path.join(folder, file)
            with open(p, 'rb') as fh:
                digest = hashlib.sha1(fh.read()).hexdigest()
            if digest in known:
                skipped += 1
                continue
            known.add(digest)
            image = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                missing.append(file)
                continue
            before = len(entries)
            if image.shape[:2] == (512, 512):
                found = get_all_faces(image)
                if not found:
                    missing.append(file)
                    continue
                _add(max(found, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])), image,
                     f"{name}: {file}", digest)
            else:
                found = extract_face_images(p, (False, 0), 0.5)
                if not found:
                    missing.append(file)
                for face, img in found:
                    _add(face, img, f"{name}: {file}", digest)
            added += [e['id'] for e in entries[before:]]
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    if added:
        _undo.append((f"add {name}", ('added', added)))
        del _undo[:-20]
    if missing:
        gr.Warning(f"No face found in {len(missing)} image(s) of {name}")
    if skipped:
        gr.Info(f"{skipped} photo(s) of {name} already in the list, not added again")
    return len(added)


def on_faceset_opened(fsz, progress=gr.Progress()):
    """Add a .fsz's photos to the list. An empty list takes the save name
    from the file."""
    if fsz is None:
        return [None, gr.Textbox()] + _render()
    path = fsz.name if hasattr(fsz, 'name') else str(fsz)
    was_empty = not entries
    added = _add_faceset_file(path, progress)
    _refresh()
    name = gr.Textbox()
    if was_empty and added:
        name = gr.Textbox(value=re.sub(r'[^\w\-]+', '_', os.path.splitext(os.path.basename(path))[0]).strip('_') or 'faceset')
    return [None, name] + _render()


def on_photo_selected(evt: gr.SelectData):
    global _selected
    if evt is not None and 0 <= evt.index < len(_view):
        _selected = _view[evt.index]
    return _detail()


def on_show_changed(value):
    global _filter, _selected
    _filter = value
    _refresh()
    if _selected is None and _view and _filter == SHOW_ATTENTION:
        _selected = _view[0]
    return _render()


def on_remove_selected():
    if _selected is None or _by_id(_selected) is None:
        return _render()
    position = _view.index(_selected) if _selected in _view else 0
    _remove({_selected}, 'remove photo')
    _refresh()
    _select_after_removal(position)
    return _render()


def on_keep_selected():
    global _selected
    e = _by_id(_selected) if _selected is not None else None
    if e is None:
        return _render()
    position = _view.index(_selected) if _selected in _view else 0
    e['kept'] = not e['kept']
    _refresh()
    if _selected not in _view:          # "Needs attention" hides it now: go on to the next one
        _select_after_removal(position)
    return _render()


def on_remove_flagged():
    global _selected
    flagged = {e['id'] for e in entries if _needs_attention(e)}
    if not flagged:
        return _render()
    if len(flagged) == len(entries):
        gr.Warning('Every photo is flagged: nothing removed. Check them one by one.')
        return _render()
    _remove(flagged, f'remove {len(flagged)} flagged')
    _refresh()
    if _selected in flagged:
        _selected = None
    gr.Info(f"Removed {len(flagged)} photo(s). Undo brings them back.")
    return _render()


def on_undo():
    global _selected
    if not _undo:
        return _render()
    label, removed = _undo.pop()
    if isinstance(removed, tuple) and removed[0] == 'added':
        # an added faceset goes out again
        gone = set(removed[1])
        entries[:] = [e for e in entries if e['id'] not in gone]
        if _selected in gone:
            _selected = None
        _refresh()
        return _render()
    for position, e in sorted(removed, key=lambda x: x[0]):
        entries.insert(min(position, len(entries)), e)
    _refresh()
    if len(removed) == 1:
        _selected = removed[0][1]['id'] if removed[0][1]['id'] in _view else _selected
    return _render()


def on_start_over():
    global _selected, _last_saved, _saved_ids
    _last_saved, _saved_ids = None, None
    if entries:
        _remove({e['id'] for e in entries}, 'start over')
    _selected = None
    _refresh()
    return [gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False),
            gr.File(value=None, visible=False), gr.Markdown('')] + _render()


def on_save(name):
    global _last_saved, _saved_ids
    if not entries:
        gr.Warning('No photos to save')
        return [gr.File(visible=False), gr.Markdown(''), gr.Button(visible=False)]
    base = re.sub(r'[^\w\-]+', '_', (name or '').strip()).strip('_') or 'faceset'
    target = os.path.join(unleashed.globals.output_path, base + '.fsz')
    n = 2
    while os.path.exists(target):
        target = os.path.join(unleashed.globals.output_path, f'{base}_{n}.fsz')
        n += 1
    folder = tempfile.mkdtemp(prefix='faceset_')
    try:
        names = []
        for k, e in enumerate(entries):
            p = os.path.join(folder, f'{k}.png')
            cv2.imwrite(p, e['image'])
            names.append(p)
        # under a temporary name first: a save cut off half way (a Colab
        # disconnect) must not leave a damaged .fsz under the real name
        part = target + '.part'
        util.zip(names, part)
        os.replace(part, target)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    flagged = sum(1 for e in entries if _needs_attention(e))
    note = f" ({flagged} still flagged)" if flagged else ''
    renamed = f" `{base}.fsz` already existed, so it was saved under a new name." if os.path.basename(target) != base + '.fsz' else ''
    _last_saved = target
    _saved_ids = [e['id'] for e in entries]
    # the file itself may be offered for download, wherever the output folder is
    gr.set_static_paths([target])
    return [gr.File(value=target, visible=True),
            gr.Markdown(f"Saved **{len(entries)} photos**{note} to `{target}`.{renamed}"),
            gr.Button(visible=True)]
