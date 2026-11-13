"""Does "Identity boost" make the swap look more like the source? (measurement)

    python tools/identity_check.py [--out DIR] [--boosts 0,0.25,0.5,0.75,1]

The user sees the opposite of what it is for: with the boost the result looks
LESS like the source. This swaps every ordered pair of the six people in
insightface's sample t1.jpg (source photo = person i, target photo = person
j; 30 pairs) through the app's own render path (faceswap_state settings ->
core.batch_process_regular, Largest face, factory settings) at each boost,
and scores every result with two face recognisers:

  buffalo_l w600k_r50   the model whose identity vector the boost pushes:
                        biased, expected to go up whatever the eye sees
  antelopev2 glintr100  another recogniser, trained on other data (Glint360K):
                        the fairer judge

For each: cosine similarity of the swapped face to the source person and to
the target person (references = the source / target photos themselves).
Prints the mean / median per boost and, per boost, in how many of the 30
pairs the similarity to the source fell compared with boost 0. Writes one
contact sheet per source person (rows = targets; columns = target | boost
0 ... 1) and summary.json to --out.
"""
import argparse
import glob
import json
import os
import sys
import time
import urllib.request
import zipfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANTELOPE_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip'


def glintr100(out):
    """Path of antelopev2's glintr100.onnx (downloaded once into out/antelopev2)."""
    folder = os.path.join(out, 'antelopev2')
    found = glob.glob(os.path.join(folder, '**', 'glintr100.onnx'), recursive=True)
    if found:
        return found[0]
    os.makedirs(folder, exist_ok=True)
    zpath = os.path.join(folder, 'antelopev2.zip')
    print(f'downloading {ANTELOPE_URL} ...', flush=True)
    urllib.request.urlretrieve(ANTELOPE_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(folder)
    os.remove(zpath)
    return glob.glob(os.path.join(folder, '**', 'glintr100.onnx'), recursive=True)[0]


def crop(img, face, size=400):
    """The face with a margin as large as itself, size x size (reflect-padded).
    The people in t1.jpg stand close together: outside the middle 70% the crop
    is blurred beyond recognition, so a neighbour's face is not taken for one
    more source / target face."""
    pad = 600
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    x0, y0, x1, y1 = face.bbox
    cx, cy, s = (x0 + x1) / 2 + pad, (y0 + y1) / 2 + pad, max(x1 - x0, y1 - y0)
    c = cv2.resize(big[int(cy - s):int(cy + s), int(cx - s):int(cx + s)], (size, size), interpolation=cv2.INTER_CUBIC)
    keep = np.zeros((size, size), np.float32)
    m = int(size * 0.15)
    keep[m:size - m, m:size - m] = 1.0
    keep = cv2.GaussianBlur(keep, (0, 0), size * 0.03)[..., None]
    blurred = cv2.GaussianBlur(c, (0, 0), size * 0.06)
    return (c * keep + blurred * (1.0 - keep)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=os.path.join(ROOT, 'temp', 'identity_check'))
    parser.add_argument('--boosts', default='0,0.25,0.5,0.75,1')
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    boosts = [float(b) for b in args.boosts.split(',')]

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import unleashed.globals as G
    from settings import Settings
    G.CFG = Settings('config.yaml')                          # read only
    G.CFG.config_file = os.path.join(out, 'config_copy.yaml')
    G.CFG.output_folder = G.output_path = os.path.join(out, 'renders')
    os.makedirs(G.output_path, exist_ok=True)
    from unleashed import core
    core.pre_check()
    G.execution_providers = core.decode_execution_providers([G.CFG.provider])
    from unleashed.face_util import get_face_analyser
    from unleashed.ProcessEntry import ProcessEntry
    from ui.tabs import faceswap_state as S
    import insightface
    from insightface.model_zoo import get_model

    analyser = get_face_analyser()
    judge = get_model(glintr100(out), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    judge.prepare(ctx_id=0)

    def face_of(img):
        faces = analyser.get(img)
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if faces else None

    def embeddings(img):
        """(w600k, glintr100) unit embeddings of the largest face, or None."""
        f = face_of(img)
        if f is None:
            return None
        a = np.asarray(f.normed_embedding, np.float32)
        b = np.asarray(judge.get(img, f), np.float32)
        return a, b / (np.linalg.norm(b) + 1e-9)

    t1 = cv2.imread(os.path.join(os.path.dirname(insightface.__file__), 'data', 'images', 't1.jpg'))
    people = sorted(analyser.get(t1), key=lambda f: f.bbox[0])
    photos, refs = [], []
    for k, f in enumerate(people):
        path = os.path.join(out, f'person{k}.png')
        img = crop(t1, f)
        cv2.imwrite(path, img)
        photos.append(path)
        refs.append(embeddings(img))
    print(f'{len(people)} people; reference embeddings: {sum(r is not None for r in refs)}', flush=True)

    # results[(src, tgt, boost)] = {'w600k_src', 'w600k_tgt', 'glint_src', 'glint_tgt', 'file'}
    results = {}
    t0 = time.time()
    for si, source in enumerate(photos):
        targets = [p for ti, p in enumerate(photos) if ti != si]
        for b in boosts:
            G.INPUT_FACESETS.clear()
            S.add_sources([source])
            S.values.update(S.FACTORY)
            S.values.update({'mode': 'Largest face', 'identity': b})
            S.apply_settings()
            v = dict(S.values)
            plugin, _ = S.mask_plugin()
            entries = [ProcessEntry(p, 0, 0, 0) for p in targets]
            core.batch_process_regular(S.SWAP_MODEL, 'File', entries, plugin, v['mask_objects'], True, None,
                                       bool(v['keep_mouth']), bool(v['keep_eyes']), int(v['passes']), None,
                                       S.active_source_index())
            for e in entries:
                ti = photos.index(e.filename)
                img = cv2.imread(e.finalname) if e.finalname and os.path.isfile(e.finalname) else None
                emb = embeddings(img) if img is not None else None
                row = {'file': e.finalname}
                if emb is not None and refs[si] is not None and refs[ti] is not None:
                    row.update(w600k_src=float(emb[0] @ refs[si][0]), w600k_tgt=float(emb[0] @ refs[ti][0]),
                               glint_src=float(emb[1] @ refs[si][1]), glint_tgt=float(emb[1] @ refs[ti][1]))
                results[(si, ti, b)] = row
        print(f'source {si} done ({time.time() - t0:.0f} s)', flush=True)

    # summary
    pairs = [(s, t) for s in range(len(photos)) for t in range(len(photos)) if s != t]
    summary = {}
    print('\nmean / median similarity of the swapped face (n pairs); "fell" = pairs whose similarity to the '
          'source is lower than at boost 0', flush=True)
    for b in boosts:
        rows = [results[(s, t, b)] for s, t in pairs if 'glint_src' in results.get((s, t, b), {})]
        base = {(s, t): results[(s, t, boosts[0])] for s, t in pairs}
        line = {'n': len(rows)}
        for k in ('w600k_src', 'w600k_tgt', 'glint_src', 'glint_tgt'):
            vals = np.array([r[k] for r in rows])
            line[k] = (round(float(vals.mean()), 4), round(float(np.median(vals)), 4)) if len(vals) else None
        for k in ('w600k_src', 'glint_src'):
            line[k + '_fell'] = sum(1 for s, t in pairs
                                    if k in results.get((s, t, b), {}) and k in base[(s, t)]
                                    and results[(s, t, b)][k] < base[(s, t)][k] - 1e-4)
        summary[b] = line
        print(f"boost {b:4.2f} (n {line['n']}): "
              f"w600k to source {line['w600k_src']} to target {line['w600k_tgt']} | "
              f"glintr100 to source {line['glint_src']} to target {line['glint_tgt']} | "
              f"fell vs boost 0: w600k {line['w600k_src_fell']}, glintr100 {line['glint_src_fell']}", flush=True)
    json.dump({'boosts': boosts, 'summary': {str(k): v for k, v in summary.items()},
               'results': {f'{s}-{t}-{b}': r for (s, t, b), r in results.items()}},
              open(os.path.join(out, 'summary.json'), 'w'), indent=1)

    # contact sheets: one per source person
    cell = 200
    for si in range(len(photos)):
        rows = []
        head = [cv2.resize(cv2.imread(photos[si]), (cell, cell))] + \
               [np.full((cell, cell, 3), 255, np.uint8) for _ in boosts]
        for k, b in enumerate(boosts):
            cv2.putText(head[k + 1], f'boost {b:g}', (10, cell // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(head[0], 'source', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(np.hstack(head))
        for ti in range(len(photos)):
            if ti == si:
                continue
            row = [cv2.resize(cv2.imread(photos[ti]), (cell, cell))]
            for b in boosts:
                r = results.get((si, ti, b), {})
                img = cv2.imread(r['file']) if r.get('file') and os.path.isfile(r['file']) else None
                img = cv2.resize(img, (cell, cell)) if img is not None else np.zeros((cell, cell, 3), np.uint8)
                if 'glint_src' in r:
                    cv2.putText(img, f"{r['glint_src']:.2f}", (8, cell - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)
                row.append(img)
            rows.append(np.hstack(row))
        cv2.imwrite(os.path.join(out, f'sheet_source{si}.jpg'), np.vstack(rows))
    print(f'\nsheets and summary.json in {out}', flush=True)


if __name__ == '__main__':
    main()
