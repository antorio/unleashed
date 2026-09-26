"""
Gradio theme + CSS + load-JS for the redesigned Unleashed UI.
Tuned against Gradio 5.9.1.

Exports:
  unleashed_theme  -> pass as theme=
  unleashed_css    -> pass as css=
  unleashed_js     -> pass as js=   (runs on app load; this is where the sticky
                                       center column lives — head=<script> does NOT
                                       reliably execute in Gradio, js= does)
"""

import gradio as gr

unleashed_theme = gr.themes.Default(
    primary_hue="orange",
    secondary_hue="orange",
    neutral_hue="gray",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    button_secondary_background_fill="white",
    button_secondary_background_fill_hover="#f9fafb",
    button_secondary_border_color="*neutral_200",
    button_secondary_border_color_hover="*neutral_300",
    button_secondary_border_color_hover_dark="*neutral_500",
    button_secondary_text_color="*neutral_700",
    # without these the dark theme reuses the light values: white buttons, text unreadable
    button_secondary_background_fill_dark="*neutral_700",
    button_secondary_background_fill_hover_dark="*neutral_600",
    button_secondary_border_color_dark="*neutral_600",
    button_secondary_text_color_dark="*neutral_100",
    block_title_text_weight="600",
    block_label_text_weight="600",
)

unleashed_css = """
/* ---------- width: fill the page ---------- */
.gradio-container { max-width: 1840px !important; width: 96% !important; margin: 0 auto !important; }

/* ---------- header: title left, versions hard right ---------- */
#app_header { padding: 4px 4px 0; border: none !important; background: transparent !important;
  justify-content: space-between !important; align-items: baseline !important; flex-wrap: nowrap !important; }
#app_header h1 { margin: 0; font-size: 19px; font-weight: 700; letter-spacing: -.01em; }
#versions { margin-left: auto !important; text-align: right !important; }
#versions, #versions * { font-family: var(--font-mono); font-size: 12.5px; color: var(--body-text-color-subdued); }

/* ---------- less chrome above the tabs (1080p screens) ---------- */
.gradio-container { padding-top: 6px !important; padding-bottom: 6px !important; }
#app_header { margin-bottom: -10px !important; }
.tabs { gap: 6px !important; }
.tabitem { padding-top: 4px !important; padding-bottom: 4px !important; }

/* ---------- tabs: bold, orange when active ---------- */
.tab-nav button, button[role="tab"], .tabs > .tab-nav button { font-weight: 700 !important; }
button.selected { color: var(--primary-600) !important; }

/* ---------- accordion / section titles: bold ---------- */
.label-wrap > span, button.label-wrap span, .gradio-accordion .label-wrap span {
  font-weight: 700 !important; font-size: 15px !important; color: var(--body-text-color) !important; }

/* ---------- hide the component-type glyph next to block labels ---------- */
.block > label > span > svg.svelte-43sxxs, span[data-testid="block-label"] svg, .block-label svg { display: none !important; }

/* ---------- sleeker secondary buttons ---------- */
button.secondary { background: var(--button-secondary-background-fill) !important; border:1px solid var(--border-color-primary) !important;
  color: var(--body-text-color) !important; box-shadow:none !important; font-weight:500 !important; font-size:13px !important; }
button.secondary:hover { background: var(--button-secondary-background-fill-hover) !important; border-color: var(--button-secondary-border-color-hover, var(--border-color-primary)) !important; }

/* ---------- shrink the Source/Target dropzones + show only icon + 'Drop File Here' ----------
   The dropzone text ('Drop File Here', '- or -', 'Click to Upload') is partly bare
   text, so we can't hide just one piece by selector. Instead: zero the wrap font
   (hides ALL its text, svg unaffected) and re-add our own single line via ::after. */
#src_drop, #tgt_files { min-height: 0 !important; }
/* Face Management: the status on each photo must stay readable */
#facemgr_gallery .caption-label { opacity: 1 !important; font-size: 12px !important; max-width: 94% !important; white-space: nowrap; text-overflow: ellipsis; }
#facemgr_gallery .thumbnail-lg:hover .caption-label { opacity: 1 !important; }
#facemgr_gallery .grid-wrap { min-height: 360px !important; max-height: 68vh !important; overflow-y: auto !important; }
#src_drop > button[tabindex], #tgt_files > button[tabindex] { height: 54px !important; min-height: 0 !important; }
#src_drop > button .wrap, #tgt_files > button .wrap { min-height: 0 !important; padding: 6px !important; font-size: 0 !important; }
#src_drop > button .wrap svg, #tgt_files > button .wrap svg { width: 18px !important; height: 18px !important; }
#src_drop > button .wrap::after, #tgt_files > button .wrap::after {
  display: block; margin-top: 2px; font-size: 12.5px; font-weight: 500; color: var(--body-text-color-subdued); }
#src_drop > button .wrap::after { content: "Drop photos or a faceset (.fsz), or click"; }
#tgt_files > button .wrap::after { content: "Drop images or videos, or click"; }
#src_drop .file-preview { min-height: 0 !important; }

/* ---------- Face Swap: lists, hints, run bar ---------- */
#src_gal .grid-container { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; }
#people_gal .grid-container { grid-template-columns: repeat(5, minmax(0, 1fr)) !important; }
#picker_gal .grid-container { grid-template-columns: repeat(6, minmax(0, 1fr)) !important; }
#picker_gal .thumbnail-item { min-height: 0 !important; cursor: pointer; }
#picker_gal .thumbnail-item:hover { outline: 3px solid var(--color-accent) !important; outline-offset: -3px; }
#picker_gal .empty { min-height: 0 !important; height: 90px !important; }
#src_gal .thumbnail-item, #people_gal .thumbnail-item { min-height: 0 !important; position: relative !important; }
#src_gal .thumbnail-item.selected, #people_gal .thumbnail-item.selected {
  outline: 3px solid var(--color-accent) !important; outline-offset: -3px; }
.fs-line, .fs-line * { font-size: 12.5px !important; line-height: 1.35 !important; }
.fs-line p { margin: 0 !important; }
#run_bar { align-items: center !important; }
#ready_line, #ready_line * { font-size: 13px !important; }
#ready_line p { margin: 0 0 2px !important; }
#status_line, #status_line * { font-size: 13px !important; }
/* while a render runs Gradio draws its progress (bar, frames, time) over the
   status line: give it room */
#status_line:has(> .wrap:not(.hide)) { min-height: 72px !important; }
#range_line, #range_line * { color: var(--body-text-color-subdued) !important; }

/* ---------- Face Swap: compartments ----------
   One box per compartment and nothing boxed inside it: Gradio draws a border
   around every block and a grey .form behind side-by-side blocks, whose gap
   showed as thick grey bars. Inside a box one spacing is used for everything;
   the boxes of a column touch; related buttons are one group. */
:root { --fs-gap: 8px; }
#swap_row { gap: 12px !important; }
#fs_left, #fs_settings { gap: 0 !important; }
.fs-box { border: 1px solid var(--border-color-primary) !important; border-radius: 0 !important;
  background: var(--block-background-fill) !important; box-shadow: none !important; padding: 10px 12px !important; }
#fs_left > .fs-box + .fs-box, #fs_settings > .fs-box + .fs-box { margin-top: -1px !important; }
.fs-box.fs-first { border-top-left-radius: 10px !important; border-top-right-radius: 10px !important; }
.fs-box.fs-last { border-bottom-left-radius: 10px !important; border-bottom-right-radius: 10px !important; }
.fs-box > .label-wrap { margin: 0 !important; }
.fs-box > .label-wrap.open { margin-bottom: var(--fs-gap) !important; }
.fs-box .block, .fs-box .form { border: none !important; box-shadow: none !important; background: transparent !important;
  border-radius: 0 !important; }
.fs-box .block { padding: 0 !important; }
.fs-box .column, .fs-box .row, .fs-box .form, #center_stage.fs-box { gap: var(--fs-gap) 12px !important; }
/* a row whose items are all hidden takes no room (it used to add an empty gap) */
.fs-box .row:not(:has(> :not(.hidden):not(.hide))) { display: none !important; }
#defaults_bar { margin-top: var(--fs-gap) !important; gap: 6px !important; }
/* lists: a soft well; drop zones: dashed; a file list: a plain frame */
.fs-box #src_gal.block, .fs-box #people_gal.block, .fs-box #picker_gal.block {
  background: var(--background-fill-secondary) !important; border-radius: 8px !important; }
.fs-box #src_drop.block, .fs-box #tgt_files.block { border: 1px dashed var(--border-color-primary) !important;
  border-radius: 8px !important; }
.fs-box #tgt_files.block:has(table) { border-style: solid !important; }
/* button groups: touching, one outline */
.fs-box .fs-seg, .fs-box .row.fs-seg { gap: 0 !important; flex-wrap: nowrap !important; }
.fs-seg > button { border-radius: 0 !important; margin-left: -1px !important; }
.fs-seg > button:first-child { border-radius: 8px 0 0 8px !important; margin-left: 0 !important; }
.fs-seg > button:last-child { border-radius: 0 8px 8px 0 !important; }
/* text fields keep their own outline (the block around them has none now) */
.fs-box input[type="text"], .fs-box textarea { border: 1px solid var(--input-border-color) !important;
  border-radius: 8px !important; background: var(--input-background-fill) !important; padding: 6px 10px !important; }
/* path + Add: one field */
.fs-box .fs-path, .fs-box .row.fs-path { gap: 0 !important; flex-wrap: nowrap !important; }
.fs-path input, .fs-path textarea { border-top-right-radius: 0 !important; border-bottom-right-radius: 0 !important; }
.fs-path > button { border-top-left-radius: 0 !important; border-bottom-left-radius: 0 !important; margin-left: -1px !important; }
.fs-buttons { gap: 6px !important; }
#view_bar, #frame_bar { align-items: center !important; }
#view_radio .wrap { gap: 4px !important; }
#view_radio label { padding: 4px 10px !important; }
#range_bar { align-items: center !important; flex-wrap: nowrap !important; }
#range_bar #range_line { flex: 1 1 auto !important; min-width: 0 !important; }
#range_bar > #range_buttons { flex: 0 0 auto !important; width: auto !important; }
#range_buttons > button { flex: 0 0 auto !important; min-width: 0 !important; padding: 0 12px !important; }
#crop_row { flex-wrap: nowrap !important; }
#crop_row > * { min-width: 0 !important; }
/* slider heads: the number always sits on the label's line (Gradio drops it
   below when the label is long, so side-by-side sliders looked different) */
/* galleries: a scrollbar only when the photos overflow (Gradio always shows a 15px track) */
.fs-box .grid-wrap { overflow-y: auto !important; }
#fs_left .head, #fs_settings .head { flex-wrap: nowrap !important; gap: 6px !important; }
#fs_left .head > label, #fs_settings .head > label { min-width: 0 !important; flex: 1 1 auto !important; }
#fs_left .head .tab-like-container, #fs_settings .head .tab-like-container { flex: 0 0 auto !important; }
/* room for 5 characters (0.125, 0.005 steps) next to the up/down arrows,
   which cover the last digit of a 48px box */
#fs_left .head input[type="number"], #fs_settings .head input[type="number"] {
  width: calc(5ch + 32px) !important; min-width: 0 !important; padding: 4px 4px 4px 6px !important; }
/* an empty gallery draws a 236px placeholder: keep the lists their own size */
#src_gal .empty { min-height: 0 !important; height: 146px !important; }
#people_gal .empty { min-height: 0 !important; height: 90px !important; }
/* the × on each source / picked person (added by unleashed_js) */
.fs-hidden { display: none !important; }
.fs-x { position: absolute; top: 3px; right: 3px; z-index: 3; width: 20px; height: 20px; border-radius: 50%;
  background: rgba(0, 0, 0, .6); color: #fff; font-size: 15px; line-height: 19px; text-align: center;
  cursor: pointer; opacity: .8; user-select: none; }
.fs-x:hover { opacity: 1; background: #dc2626; }
/* "One source per face": the order, small in a corner (no captions over the faces) */
#src_gal.fs-numbered .grid-container { counter-reset: fs-src; }
#src_gal.fs-numbered .thumbnail-item { counter-increment: fs-src; }
#src_gal.fs-numbered .thumbnail-item::before { content: counter(fs-src); position: absolute; top: 3px; left: 3px; z-index: 3;
  min-width: 18px; height: 18px; padding: 0 4px; border-radius: 9px; background: var(--color-accent); color: #fff;
  font-size: 11px; font-weight: 700; line-height: 18px; text-align: center; }
/* target list: the file shown in the preview is marked (unleashed_js) */
#tgt_files tr.file { cursor: pointer; }
#tgt_files tr.file.fs-shown { background: var(--color-accent-soft) !important; box-shadow: inset 3px 0 0 var(--color-accent); }
#tgt_files tr.file.fs-shown .stem { font-weight: 600; }

/* preview badge: updating / updated (unleashed_js); over the picture, no layout */
#preview_img { position: relative; }
.fs-pbadge { position: absolute; left: 10px; bottom: 10px; z-index: 6; padding: 3px 9px; border-radius: 999px;
  background: rgba(17, 24, 39, .72); color: #fff; font-size: 12px; line-height: 18px; white-space: nowrap;
  pointer-events: auto; user-select: none; -webkit-user-select: none; transition: opacity .6s ease;
  max-width: calc(100% - 20px); overflow: hidden; text-overflow: ellipsis; }
.fs-pbadge:empty, .fs-pbadge.fs-off { display: none; }
.fs-pbadge.fs-busy::before { content: ""; display: inline-block; width: 9px; height: 9px; margin-right: 6px;
  border: 2px solid rgba(255, 255, 255, .35); border-top-color: #fff; border-radius: 50%;
  vertical-align: -1px; animation: fs-spin .8s linear infinite; }
@keyframes fs-spin { to { transform: rotate(360deg); } }
.fs-pbadge.fs-done { background: rgba(21, 128, 61, .85); }
.fs-pbadge.fs-warn { background: rgba(180, 83, 9, .88); }
.fs-pbadge.fs-dim { opacity: .45; }
.fs-pbadge.fs-dim:hover { opacity: 1; }
.fs-pbadge.fs-can-compare { cursor: pointer; }
.fs-pprev { position: absolute; z-index: 5; object-fit: contain; pointer-events: none; }
.fs-checks label { white-space: nowrap !important; }
/* Gradio dims a Markdown to 20% while any event writing it runs; the readiness
   line is refreshed by every preview, so it was faded most of the time */
#ready_line .pending, #status_line .pending, #fs_left .pending { opacity: 1 !important; }
#ready_line code, #status_line code { white-space: normal; word-break: break-all; }
#mask_editor button[aria-label="Clear canvas"] { display: none !important; }

/* ---------- Eyes / Mouth / Brows forced onto a single row ----------
   Gradio groups the 3 adjacent checkboxes into a .form wrapper that wraps at 2.
   Force that .form (and its children) to a single nowrap flex row. */
#expr_checks .form, #expr_checks > div {
  display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important; gap: 12px !important; }
#expr_checks .form > *, #expr_checks > div > * { flex: 1 1 0 !important; min-width: 0 !important; }
#expr_checks label { white-space: nowrap !important; }

/* ---------- centre column stays in view while the settings scroll ----------
   Pure CSS sticky (the container's overflow:hidden blocked it before). Only
   where the three columns fit side by side; the column scrolls on its own
   when it is taller than the window. */
.gradio-container { overflow: unset !important; }
#swap_row { align-items: flex-start !important; }
@media (min-width: 1200px) {
  #swap_row { flex-wrap: nowrap !important; }
  #center_stage { position: sticky; top: 8px; align-self: flex-start; max-height: calc(100vh - 16px);
                  overflow-y: auto; flex-wrap: nowrap !important; }
  #center_stage > * { flex-shrink: 0 !important; }
}

/* ---------- tidy spacing ---------- */
.block { border-radius: 8px; }

/* ---------- clean footer: hide Gradio default (Use via API · Built with Gradio ·
   Settings + icons), show a single tidy 'Use via API' line ---------- */
footer { display: none !important; }
.rl-footer {
    text-align: center; padding: 20px; font-size: 11px;
    color: var(--body-text-color-subdued); font-family: var(--font-mono);
}
"""

# Runs on app load (gr.Blocks(js=...)). The centre column used to follow the
# scroll with a requestAnimationFrame loop; CSS sticky does it now.
unleashed_js = """
() => {
    // Face Swap: the arrow keys step the preview frame (like its ◀ ▶ buttons)
    // unless the focus is in a text field or a slider being dragged.
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
        const el = document.activeElement;
        // taken over as well: the frame slider (its own arrow stepping does not
        // refresh the preview) and the View radios (a click leaves the focus
        // there, and arrows would switch the view instead of the frame)
        const ours = el && el.tagName === 'INPUT' && el.closest &&
            ((el.type === 'range' && el.closest('#frame_slider')) || (el.type === 'radio' && el.closest('#view_radio')));
        if (!ours && el && (el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName))) return;
        const btn = document.getElementById(e.key === 'ArrowLeft' ? 'frame_prev' : 'frame_next');
        if (!btn || !btn.offsetParent) return;          // other tab, or not a video
        e.preventDefault();
        btn.click();
    });

    // The path boxes start with a folder (Settings): after the click (or Tab)
    // that focuses one, typing continues after that folder, wherever the
    // click landed. The caret is moved on the click and again right before the
    // first key or text goes in (a timer would lose to fast typing). A click
    // in a box that already has the focus, or an arrow key, is left alone.
    const pathBox = (t) => (t && t.closest && ['INPUT', 'TEXTAREA'].includes(t.tagName)
                            && t.closest('#src_path, #tgt_path')) ? t : null;
    const toEnd = (el) => {
        const n = el.value.length;
        try { el.setSelectionRange(n, n); } catch (err) {}
        el.scrollLeft = el.scrollWidth;
    };
    const endOnce = (el) => {
        if (!el || el.dataset.fsEnd !== '1') return;
        delete el.dataset.fsEnd;
        toEnd(el);
    };
    document.addEventListener('mousedown', (e) => {
        const el = pathBox(e.target);
        if (!el) return;
        if (document.activeElement !== el) el.dataset.fsEnd = '1';
        else delete el.dataset.fsEnd;
    }, true);
    document.addEventListener('click', (e) => {
        const el = pathBox(e.target);
        if (!el || el.dataset.fsEnd !== '1') return;
        if (el.selectionStart === el.selectionEnd) toEnd(el);
        else delete el.dataset.fsEnd;       // the focusing click dragged a selection: keep it
    });
    document.addEventListener('keydown', (e) => {
        const el = pathBox(e.target);
        if (!el) return;
        if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) delete el.dataset.fsEnd;
        else endOnce(el);
    }, true);
    document.addEventListener('beforeinput', (e) => endOnce(pathBox(e.target)), true);
    // Tab into a box (Chrome would select the whole path); not when the box
    // only gets its focus back with the window (or the Colab frame)
    let windowBlurredOn = null;
    window.addEventListener('blur', () => { windowBlurredOn = document.activeElement; });
    document.addEventListener('focusin', (e) => {
        const el = pathBox(e.target);
        const refocus = el && el === windowBlurredOn;
        windowBlurredOn = null;
        if (!el || refocus || el.dataset.fsEnd === '1') return;
        el.dataset.fsEnd = '1';             // fast typing beats the timer: keydown moves it too
        setTimeout(() => toEnd(el), 0);
    });
    document.addEventListener('focusout', (e) => {
        const el = pathBox(e.target);
        if (el) delete el.dataset.fsEnd;
    });

    // × on each source and picked-person thumbnail. A click on it sends
    // "index:time" through a hidden textbox (its .input event runs the
    // removal on the server) and never reaches the thumbnail (no select).
    const X_TARGETS = [['src_gal', 'src_x'], ['people_gal', 'people_x']];
    document.addEventListener('click', (e) => {
        const x = e.target && e.target.closest ? e.target.closest('.fs-x') : null;
        if (!x) return;
        e.preventDefault();
        e.stopPropagation();
        const gallery = document.getElementById(x.dataset.gallery);
        const item = x.closest('.thumbnail-item');
        if (!gallery || !item) return;
        const index = [...gallery.querySelectorAll('.thumbnail-item')].indexOf(item);
        const box = document.querySelector('#' + x.dataset.box + ' textarea, #' + x.dataset.box + ' input');
        if (index < 0 || !box) return;
        box.value = index + ':' + Date.now();
        box.dispatchEvent(new Event('input', {bubbles: true}));
    }, true);

    const tidy = () => {
        for (const [gid, bid] of X_TARGETS) {
            const gallery = document.getElementById(gid);
            if (!gallery) continue;
            gallery.querySelectorAll('.thumbnail-item').forEach((item) => {
                if (item.querySelector(':scope > .fs-x')) return;
                const x = document.createElement('span');
                x.className = 'fs-x';
                x.textContent = '×';
                x.title = 'Remove';
                x.dataset.gallery = gid;
                x.dataset.box = bid;
                item.appendChild(x);
            });
        }
        // order numbers only when the order matters
        const src = document.getElementById('src_gal');
        const mode = document.querySelector('#mode_dd input');
        if (src && mode) src.classList.toggle('fs-numbered', (mode.value || '').startsWith('One source'));
        // mark the target file the preview shows (its name: a hidden field)
        const field = document.querySelector('#tgt_shown textarea, #tgt_shown input');
        const shown = field ? field.value : '';
        document.querySelectorAll('#tgt_files tr.file').forEach((row) => {
            const cell = row.querySelector('td.filename');
            row.classList.toggle('fs-shown', !!cell && cell.getAttribute('aria-label') === shown);
        });
    };
    let queued = false;
    new MutationObserver(() => {
        if (queued) return;
        queued = true;
        setTimeout(() => { queued = false; tidy(); }, 60);
    }).observe(document.body, {childList: true, subtree: true, characterData: true});
    setInterval(tidy, 1000);        // the dropdown's value is no DOM mutation

    // Preview badge (bottom left of the preview): "Updating… n s" from the
    // moment a change asks for a new preview (the hidden tick moves, or
    // Refresh) until the answer for the LATEST tick arrives, then "Updated in
    // n s" and how the picture differs from the previous one of the same
    // file, frame and view (measured on the server: "no change", "n %
    // changed"). It then dims and stays; press and hold it to see the previous
    // picture.
    const val = (sel) => { const el = document.querySelector(sel); return el ? el.value : null; };
    const pv = {tick: null, answer: null, pending: false, t0: 0, dimTimer: null, shown: []};
    const shownImage = () => {
        const img = document.querySelector('#preview_img img[src]');
        return img && img.complete && img.naturalWidth ? img : null;
    };
    const badge = () => {
        const block = document.getElementById('preview_img');
        if (!block) return null;
        let b = block.querySelector(':scope > .fs-pbadge');
        if (!b) {
            b = document.createElement('div');
            b.className = 'fs-pbadge fs-off';
            block.appendChild(b);
            const hold = (on) => (e) => {
                block.querySelectorAll('.fs-pprev').forEach((x) => x.remove());
                if (!on || !b.classList.contains('fs-can-compare') || pv.shown.length < 2) return;
                e.preventDefault();
                const img = shownImage();
                if (!img) return;
                const r = img.getBoundingClientRect(), br = block.getBoundingClientRect();
                const o = document.createElement('img');
                o.className = 'fs-pprev';
                o.src = pv.shown[pv.shown.length - 2];
                // the same box AND the same fitting as the picture shown: the
                // element is often larger than the picture in it (a wide Side
                // by side picture in a tall box), and 'fill' stretched it
                const cs = getComputedStyle(img);
                Object.assign(o.style, {left: (r.left - br.left) + 'px', top: (r.top - br.top) + 'px',
                                        width: r.width + 'px', height: r.height + 'px',
                                        objectFit: cs.objectFit || 'contain', objectPosition: cs.objectPosition});
                block.appendChild(o);
            };
            b.addEventListener('pointerdown', hold(true));
            ['pointerup', 'pointerleave', 'pointercancel'].forEach((t) => b.addEventListener(t, hold(false)));
        }
        return b;
    };
    const say = (text, cls, title) => {
        const b = badge();
        if (!b) return;
        b.textContent = text;
        b.title = title || '';
        b.className = 'fs-pbadge ' + (cls || '');
    };
    const finish = (state, change) => {
        clearTimeout(pv.dimTimer);
        const secs = ((performance.now() - pv.t0) / 1000).toFixed(1);
        if (state === 'failed') return say('⚠ Preview failed', 'fs-warn');
        if (state === 'paused') return say('Paused while rendering', 'fs-warn');
        if (state !== 'shown') return say('', 'fs-off');          // no picture, auto-update off, painting
        const f = Number(change);
        let text = '✓ Updated in ' + secs + ' s', cls = 'fs-done', title = '';
        if (change === 'identical') text += ' · no change';
        else if (change === 'tiny') text += ' · tiny change (1–2 levels)';
        else if (change && f > 0) {
            text += ' · ' + (f < 0.001 ? '<0.1' : (f * 100).toFixed(1)) + '% changed · hold to compare';
            cls += ' fs-can-compare';
            title = 'Press and hold to see the previous picture';
        }
        say(text, cls, title);
        pv.dimTimer = setTimeout(() => { const b = badge(); if (b) b.classList.add('fs-dim'); }, 4000);
    };
    const watch = () => {
        // the pictures shown, newest last (the one before the newest is "previous")
        const img = shownImage();
        const src = img ? (img.currentSrc || img.src) : null;
        if (src && pv.shown[pv.shown.length - 1] !== src) pv.shown = pv.shown.concat([src]).slice(-3);
        const tick = val('#fs_tick input'), answer = val('#fs_preview_done textarea, #fs_preview_done input');
        if (tick === null) return;
        if (pv.tick === null) { pv.tick = tick; pv.answer = answer; return; }
        if (tick !== pv.tick) {
            pv.tick = tick;
            const auto = document.querySelector('#fs_auto input[type=checkbox]');
            const view = val('#view_radio input:checked');
            if (!pv.pending && ((auto && auto.checked) || view === 'Original' || view === 'Mask')) {
                pv.pending = true;
                pv.t0 = performance.now();
            }
        }
        if (answer !== pv.answer) {
            pv.answer = answer;
            const [forTick, state, , change] = (answer || '').split('|');
            // an answer for an older tick (a preview that was already running): keep waiting
            if (!pv.pending || Number(forTick) === Number(pv.tick)) {
                if (!pv.pending) pv.t0 = performance.now();
                pv.pending = false;
                finish(state, change || '');
            }
        }
        if (pv.pending) {
            clearTimeout(pv.dimTimer);
            say('Updating preview… ' + Math.floor((performance.now() - pv.t0) / 1000) + ' s', 'fs-busy');
        }
    };
    document.addEventListener('click', (e) => {
        if (e.target && e.target.closest && e.target.closest('#fs_refresh') && !pv.pending) {
            pv.pending = true;
            pv.t0 = performance.now();
        }
    }, true);
    setInterval(watch, 150);
}
"""
