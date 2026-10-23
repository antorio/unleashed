import threading
import types
from typing import Any
import insightface

import unleashed.globals
from unleashed.typing import Frame, Face

import cv2
import numpy as np
from skimage import transform as trans
from unleashed.capturer import get_video_frame
from unleashed.utilities import resolve_relative_path, conditional_thread_semaphore

FACE_ANALYSER = None
# Building (and on first use downloading) buffalo_l must happen once. The old
# guard was conditional_thread_semaphore(), which is a no-op on CUDA, so two
# Gradio events at the same moment both started the download and the second
# crashed with FileExistsError on models/buffalo_l.
_ANALYSER_LOCK = threading.Lock()
_ANALYSER_CPU = None        # force_cpu the current analyser was built with
FACE_SWAPPER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER, _ANALYSER_CPU

    with _ANALYSER_LOCK:
        force_cpu = bool(unleashed.globals.CFG.force_cpu)
        if (FACE_ANALYSER is None or _ANALYSER_CPU != force_cpu
                or unleashed.globals.g_current_face_analysis != unleashed.globals.g_desired_face_analysis):
            _ANALYSER_CPU = force_cpu
            model_path = resolve_relative_path('..')
            # removed genderage
            allowed_modules = unleashed.globals.g_desired_face_analysis
            unleashed.globals.g_current_face_analysis = unleashed.globals.g_desired_face_analysis
            if unleashed.globals.CFG.force_cpu:
                print("Forcing CPU for Face Analysis")
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    root=model_path, providers=["CPUExecutionProvider"],allowed_modules=allowed_modules
                )
            else:
                from unleashed.utilities import tuned_execution_providers
                provs = tuned_execution_providers()
                print(f"[load] Building buffalo_l face analyser  modules={allowed_modules}  providers={provs}")
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name="buffalo_l", root=model_path, providers=provs,allowed_modules=allowed_modules
                )
            if insightface.__version__ == '0.7.3':      # _analyse copies its get()
                FACE_ANALYSER.get = types.MethodType(_analyse, FACE_ANALYSER)
            unleashed.globals.g_current_det_params = None
        # Detector settings are live: prepare() is re-run whenever the threshold
        # or the detection size changes, so moving the sliders takes effect
        # without restarting (it only re-configures the session, no reload).
        det_size = int(getattr(unleashed.globals, 'det_size', 640) or 640)
        det_thresh = float(getattr(unleashed.globals, 'det_thresh', 0.5) or 0.5)
        params = (det_size, det_thresh)
        if getattr(unleashed.globals, 'g_current_det_params', None) != params:
            FACE_ANALYSER.prepare(ctx_id=0, det_thresh=det_thresh,
                                  det_size=(det_size, det_size))
            unleashed.globals.g_current_det_params = params
    return FACE_ANALYSER


def _analyse(self, img, max_num=0):
    """FaceAnalysis.get of insightface 0.7.3 line for line, except that the
    gender/age model runs only when a swap mode reads face.sex (All women /
    All men: unleashed.globals.face_gender_needed). Every face of every frame
    ran it; nothing else reads gender or age. It stays loaded (no rebuild
    when the mode changes). Each model of the analysis works on its own, so
    the other results are the same either way. Fewer runs of it also mean
    fewer chances for its rare cuDNN errors (onnx_guard)."""
    bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')
    if bboxes.shape[0] == 0:
        return []
    skip = ('detection',) if getattr(unleashed.globals, 'face_gender_needed', False) else ('detection', 'genderage')
    ret = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        for taskname, model in self.models.items():
            if taskname in skip:
                continue
            model.get(img, face)
        ret.append(face)
    return ret


def get_first_face(frame: Frame) -> Any:
    try:
        faces = get_face_analyser().get(frame)
        # Pick the largest face (main subject) rather than the left-most one.
        # The left-most heuristic frequently latched onto a small background
        # face; largest-area is what "first face" almost always means.
        return max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    except:
        return None


def get_all_faces(frame: Frame) -> Any:
    try:
        faces = get_face_analyser().get(frame)
        return sorted(faces, key=lambda x: x.bbox[0])
    except:
        return None


# ---------------------------------------------------------------------------
# Multi-angle detection: rotate the frame by 0/90/180/270 so the detector and
# landmark models see a near-upright face, then map every detected face back to
# the original frame coordinates. The 90-degree rotations are lossless, so the
# mapped landmarks are exact. Modes: 'off', 'fallback' (only rotate when 0 deg
# finds nothing), 'always' (union over all angles).
# Point back-transforms were verified empirically against cv2.rotate.
# ---------------------------------------------------------------------------

def _rotate_frame_for_angle(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _unrotate_points(pts, angle, H, W):
    # pts are in the rotated frame; return coordinates in the original HxW frame
    pts = np.asarray(pts, dtype=np.float32)
    x = pts[..., 0]
    y = pts[..., 1]
    if angle == 90:
        ox, oy = y, (H - 1) - x
    elif angle == 180:
        ox, oy = (W - 1) - x, (H - 1) - y
    elif angle == 270:
        ox, oy = (W - 1) - y, x
    else:
        return pts.copy()
    return np.stack([ox, oy], axis=-1).astype(np.float32)


def _transform_face_to_original(face, angle, H, W):
    x1, y1, x2, y2 = [float(v) for v in face['bbox']]
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    oc = _unrotate_points(corners, angle, H, W)
    face['bbox'] = np.array([oc[:, 0].min(), oc[:, 1].min(),
                             oc[:, 0].max(), oc[:, 1].max()], dtype=np.float32)
    kps = face.get('kps', None)
    if kps is not None:
        face['kps'] = _unrotate_points(kps, angle, H, W)
    lm106 = face.get('landmark_2d_106', None)
    if lm106 is not None:
        face['landmark_2d_106'] = _unrotate_points(lm106, angle, H, W)
    lm68 = face.get('landmark_3d_68', None)
    if lm68 is not None:
        lm68 = np.asarray(lm68).copy()
        lm68[:, :2] = _unrotate_points(lm68[:, :2], angle, H, W)
        face['landmark_3d_68'] = lm68
    return face


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _dedup_faces(collected, iou_thr=0.5):
    """Drop overlapping detections of the same face, keeping the best one.

    Angle-0 (upright) detections get a score bonus, `unleashed.globals.angle0_bonus`:
      1.0 (default) = angle-0 ALWAYS wins, whatever the scores. Original
                      behaviour: the footage is upright, so the upright pass is
                      treated as the truth and rotated passes are only a
                      fallback for faces that are sideways in the frame.
      0.10          = angle-0 wins ties and near-ties, but a clearly more
                      confident rotated detection wins on merit. On a sideways
                      head the upright pass often scrapes past the threshold
                      with poor keypoints on some frames and fails on others --
                      absolute priority makes the alignment flip between the two
                      every few frames even though the head barely moved.
      0.0           = pure score, no preference.
    Ties always fall to angle-0.
    """
    bonus = float(getattr(unleashed.globals, 'angle0_bonus', 1.0))

    def key(item):
        ang, f = item
        s = float(getattr(f, 'det_score', 0.0) or 0.0)
        if ang == 0:
            s += bonus
        return (-s, 0 if ang == 0 else 1)
    kept = []
    for ang, f in sorted(collected, key=key):
        if any(_bbox_iou(f.bbox, kf.bbox) > iou_thr for kf in kept):
            continue
        kept.append(f)
    return kept


def get_all_faces_multi(frame: Frame, mode='fallback', angles=None) -> Any:
    if mode == 'off' or mode is None:
        return get_all_faces(frame)
    if angles is None:
        angles = [90, 270, 180]
    analyser = get_face_analyser()
    H, W = frame.shape[:2]

    try:
        with conditional_thread_semaphore():
            base = analyser.get(frame)
    except Exception:
        base = []
    base = base or []

    if mode == 'fallback' and len(base) > 0:
        return sorted(base, key=lambda x: x.bbox[0])

    collected = [(0, f) for f in base]
    for ang in angles:
        rf = _rotate_frame_for_angle(frame, ang)
        try:
            with conditional_thread_semaphore():
                faces = analyser.get(rf)
        except Exception:
            faces = None
        if not faces:
            continue
        for f in faces:
            collected.append((ang, _transform_face_to_original(f, ang, H, W)))
        if mode == 'fallback' and len(collected) > 0:
            break

    if len(collected) == 0:
        return None
    kept = _dedup_faces(collected)
    return sorted(kept, key=lambda x: x.bbox[0])


def get_first_face_multi(frame: Frame, mode='fallback', angles=None) -> Any:
    faces = get_all_faces_multi(frame, mode=mode, angles=angles)
    if not faces:
        return None
    return max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))


def extract_face_images(source_filename, video_info, extra_padding=-1.0, use_multi_angle=False):
    """Collect the faces of an image / video frame for the UI pickers.

    use_multi_angle: run the SAME rotated-detection passes the swap uses. Off by
    default -- source photos and the face manager are upright, so trying four
    orientations there only costs time and can invent faces from odd angles.
    It is turned on for picking a TARGET face out of a video frame, because the
    swap itself detects with rotation: without it a ~90-degree face reports "no
    face detected" in the picker while the render finds and swaps it, so it can
    never be marked as a target for "selected" mode.
    """
    face_data = []
    source_image = None

    if video_info[0]:
        frame = get_video_frame(source_filename, video_info[1], exact=True)
        if frame is not None:
            source_image = frame
        else:
            return face_data
    else:
        source_image = cv2.imdecode(np.fromfile(source_filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    if use_multi_angle:
        faces = get_all_faces_multi(
            source_image,
            mode=getattr(unleashed.globals, 'multi_angle_detection_mode', 'fallback'))
    else:
        faces = get_all_faces(source_image)
    if faces is None:
        return face_data

    if extra_padding > 0.0 and source_image.shape[:2] == (512, 512) and len(faces) > 0:
        # already a 512 cut-out (a faceset PNG): kept whole, once, with its
        # largest face -- not once per face in it
        main = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return [[main, source_image]]

    i = 0
    for face in faces:
        (startX, startY, endX, endY) = face["bbox"].astype("int")
        startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
        if extra_padding > 0.0:

            found = False
            for i in range(1, 3):
                (startX, startY, endX, endY) = face["bbox"].astype("int")
                startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
                cutout_padding = extra_padding
                # top needs extra room for detection
                padding = int((endY - startY) * cutout_padding)
                oldY = startY
                startY -= padding

                factor = 0.25 if i == 1 else 0.5
                cutout_padding = factor
                padding = int((endY - oldY) * cutout_padding)
                endY += padding
                padding = int((endX - startX) * cutout_padding)
                startX -= padding
                endX += padding
                startX, endX, startY, endY = clamp_cut_values(
                    startX, endX, startY, endY, source_image
                )
                face_temp = source_image[startY:endY, startX:endX]
                face_temp = resize_image_keep_content(face_temp)
                testfaces = get_all_faces(face_temp)
                if testfaces is not None and len(testfaces) > 0:
                    i += 1
                    # the face this cut-out was made for is the largest one in
                    # it (testfaces is sorted left to right, so [0] could be
                    # part of a neighbour); remember its size in the photo
                    main = max(testfaces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    bw, bh = face["bbox"][2] - face["bbox"][0], face["bbox"][3] - face["bbox"][1]
                    main['source_size'] = float(max(bw, bh))
                    face_data.append([main, face_temp])
                    found = True
                    break

            if not found:
                print("No face found after resizing, this shouldn't happen!")
            continue

        face_temp = source_image[startY:endY, startX:endX]
        if face_temp.size < 1:
            continue

        i += 1
        face_data.append([face, face_temp])
    return face_data


def clamp_cut_values(startX, endX, startY, endY, image):
    if startX < 0:
        startX = 0
    if endX > image.shape[1]:
        endX = image.shape[1]
    if startY < 0:
        startY = 0
    if endY > image.shape[0]:
        endY = image.shape[0]
    return startX, endX, startY, endY



def resize_image_keep_content(image, new_width=512, new_height=512):
    dim = None
    (h, w) = image.shape[:2]
    if h > w:
        r = new_height / float(h)
        dim = (int(w * r), new_height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = new_width / float(w)
        dim = (new_width, int(h * r))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    (h, w) = image.shape[:2]
    if h == new_height and w == new_width:
        return image
    resize_img = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    offs = (new_width - w) if h == new_height else (new_height - h)
    startoffs = int(offs // 2) if offs % 2 == 0 else int(offs // 2) + 1
    offs = int(offs // 2)

    if h == new_height:
        resize_img[0:new_height, startoffs : new_width - offs] = image
    else:
        resize_img[startoffs : new_height - offs, 0:new_width] = image
    return resize_img


def rotate_image_90(image, rotate=True):
    if rotate:
        return np.rot90(image)
    else:
        return np.rot90(image, 1, (1, 0))


def rotate_anticlockwise(frame):
    return rotate_image_90(frame)


def rotate_clockwise(frame):
    return rotate_image_90(frame, False)


# alignment code from insightface https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


""" def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    elif image_size % 512 == 0:
        ratio = float(image_size) / 512.0
        diff_x = 32.0 * ratio

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M
 """

def estimate_norm(lmk, image_size=112):
    """Standard arcface 5-point alignment template estimation.

    The previous version added hard-coded magic offsets (+0.1/+0.5/+0.75/+1.5)
    that were only approximately right. ReSwapper documented the exact sub-pixel
    correction needed when warping to a resolution other than 112/128:
        offset = (128/32768) * resolution - 0.5
    which gives 160->0.125, 256->0.5, 320->0.75, 512->1.5. Using the formula
    removes the residual misalignment that compounds at extreme angles.
    """
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    if image_size != 112 and image_size != 128:
        offset = (128.0 / 32768.0) * image_size - 0.5
        dst[:, 0] += offset
        dst[:, 1] += offset

    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def _arcface_dst_for_size(image_size):
    """Return the arcface destination template scaled to image_size."""
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    if image_size != 112 and image_size != 128:
        offset = (128.0 / 32768.0) * image_size - 0.5
        dst[:, 0] += offset
        dst[:, 1] += offset
    return dst.astype(np.float32)


def estimate_norm_robust(lmk, image_size=128):
    """Fit the alignment matrix with a RANSAC affine (4-DOF, like a similarity
    transform but robust to one or two bad landmarks). This matches FaceFusion's
    approach and is markedly more stable than plain least-squares at profile /
    looking-up / looking-down poses where a single keypoint can be off.
    """
    dst = _arcface_dst_for_size(image_size)
    src = np.asarray(lmk, dtype=np.float32).reshape(5, 2)
    M, _ = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=100
    )
    if M is None:
        # Robust fit failed (degenerate points) -> fall back to similarity.
        return estimate_norm(lmk, image_size)
    return M.astype(np.float64)


# Standard, well-established 68 -> 5 keypoint mapping.
# left eye center, right eye center, nose tip, left mouth corner, right mouth corner
def landmark_68_to_5(landmark_68):
    lm = np.asarray(landmark_68, dtype=np.float32)[:, :2]
    left_eye = lm[36:42].mean(axis=0)
    right_eye = lm[42:48].mean(axis=0)
    nose = lm[30]
    mouth_left = lm[48]
    mouth_right = lm[54]
    return np.stack([left_eye, right_eye, nose, mouth_left, mouth_right]).astype(np.float32)



# aligned, M = norm_crop2(f[1], face.kps, 512)
def align_crop(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size)
    # BORDER_REPLICATE, not a black borderValue: when the aligned crop reaches
    # past the edge of the frame the missing area used to be filled with black.
    # That black then travelled all the way through the swap and, on paste-back,
    # the Gaussian-blurred matte spreads slightly BEYOND the crop quad where
    # BORDER_REPLICATE copies the crop's edge pixel -- i.e. the black -- leaving
    # a dark line along the swap box. It is worse at small subsample sizes
    # because the crop is magnified more on the way back, widening the band.
    # Replicating the edge keeps real skin tones there instead.
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderMode=cv2.BORDER_REPLICATE)
    return warped, M


def align_crop_robust(img, landmark5, image_size=128):
    """Align using a RANSAC affine fit. landmark5 is the 5-point set (kps or
    a 68->5 derived set). Returns the warped crop and the 2x3 matrix."""
    M = estimate_norm_robust(landmark5, image_size)
    # see align_crop: replicate instead of black-filling out-of-frame area
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderMode=cv2.BORDER_REPLICATE)
    return warped, M


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    # see align_crop: replicate instead of black-filling out-of-frame area
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderMode=cv2.BORDER_REPLICATE)
    return cropped, M


def create_blank_image(width, height):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:] = [0,0,0,0]
    return img

