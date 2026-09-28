from settings import Settings
from typing import List

source_path = None
target_path = None
output_path = None
target_folder_path = None
startup_args = None

cuda_device_id = 0
frame_processors: List[str] = []
keep_fps = None
keep_frames = None
autorotate_faces = None
vr_mode = None
skip_audio = None
wait_after_extraction = None
use_batch = None
source_face_index = 0
target_face_index = 0
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
selected_enhancer = None
subsample_size = 128
# Mask edge treatment. GLOBAL, not per source face: these describe how the swap
# is blended into the target, not a property of the source. They used to live in
# each faceset's mask_offsets, but the sliders never reloaded when you switched
# source face -- so they showed the last value you dragged regardless of which
# face was selected, and silently did nothing when no source face was loaded.
mask_erosion_iterations = 1     # how much the matte is shrunk (area)
mask_blur_size = 20             # edge softness only (does not shrink the area)
# Bottom edge of the matte (A/B toggle; False = the old behaviour). The arcface
# crop puts the chin at ~87-92% of its height, so Erosion / Blur size, applied
# to all four sides of the face square, could leave the chin unswapped: a strong
# target chin showing through, often with a visible line across it. True: the
# bottom edge ignores Erosion / Blur size and fades out over the last 6% of the
# crop instead; the top and side edges are unchanged.
# Default ON since 26 Sep (user).
mask_bottom_to_chin = True
# Build the paste matte in the face crop's own coordinates (A/B toggle; False =
# the old frame-pixel erosion). The old erosion used an axis-aligned kernel sized
# from the bounding box, so tilted faces were eroded ~75% deeper at 25 degrees.
# Default ON since 26 Sep (user).
mask_face_aligned = True
# The occlusion mask's own edge (Occlusion box; 0 = the mask exactly as the
# model draws it). With a mask model on, the edge of the swapped face is the
# mask's edge -- Erosion / Blur above only shape the face square around it.
# Both in percent of the face crop, so they do the same at every resolution.
#   grow:   + the swapped face reaches further out (over the target's own jaw
#           line / hairline), - more of the edge stays original
#   soften: feather (Gaussian) of that edge
occlusion_mask_grow = 0.0
occlusion_mask_soften = 0.0
# Passes > 1: bring every later pass back to the first pass's size, position
# and colour (Swap box; False = each pass simply feeds the previous one's
# output, and the drift adds up -- see ProcessMgr.swap_passes_keeping_look).
passes_keep_look = False

# Identity strength, 0-1 (0 = off): push the source identity away from the
# target's own identity before it conditions inswapper (see FaceSwapInsightFace).
identity_strength = 0.0
# Face shape from the source, 0-1 (0 = off): warp the target's jaw / chin toward
# the source's before swapping (see unleashed/face_shape.py). Runs buffalo_l's 68-pt
# 3D landmark model on every face while on.
face_shape_strength = 0.0
face_swap_mode = None
blend_ratio = 0.5
# Specific people: max cosine distance for a face to count as a picked person
# (user default 0.8, 26 Sep; was 0.65). ProcessMgr.match_selected_faces.
distance_threshold = 0.8
# Face detector settings (live -- see get_face_analyser).
# det_thresh: confidence a detection must reach to be accepted. insightface's
# default is 0.5; faces sitting right at the threshold flicker in and out
# between frames (swapped / not swapped / swapped), so lowering it keeps
# borderline faces (profile, motion blur, small) swapped consistently.
det_thresh = 0.5
# det_size: detector input resolution. Larger finds smaller/farther faces at
# the cost of speed; 320 starts missing faces below roughly 36px.
det_size = 640
# How strongly an upright (angle-0) detection is preferred over a rotated one
# when both cover the same face. 1.0 = upright always wins (original), 0.10 =
# only wins ties and near-ties, 0.0 = pure detector score.
angle0_bonus = 1.0

# --- Accuracy / quality toggles (Unleashed improvements) ---
# Derive the 5 alignment keypoints from the stable 68-point landmark model
# (instead of the detector's raw kps) and fit the warp with a RANSAC affine.
# This is the main fix for "off" swaps at extreme yaw/pitch angles.
use_landmark_alignment = True
# Optional Reinhard (LAB) color transfer of the swapped face toward the
# target region before paste-back. Off by default (inswapper already does
# reasonable color); enable if source/target lighting differs a lot.
use_color_transfer = False
# Occlusion mask ordering: the engine mask (Clip2Seg/XSeg/FaceParser) always runs
# after the expression restorer now. False = mask runs before the enhancer;
# True (default) = mask runs after the enhancer too, so restored occluders
# (hands/hair/etc) are not altered by the enhancer.
mask_after_enhancer = True

# Multi-angle detection: try rotated copies of the frame so sideways / upside
# down faces are found and landmarked upright. 'off' | 'fallback' | 'always'.
# 'fallback' (default) only rotates when 0 deg finds nothing -> nearly free.
# 'always' unions all angles (max recall, ~4x detection cost).
multi_angle_detection_mode = 'fallback'
multi_angle_angles = [90, 270, 180]

# Temporal landmark smoothing for video (reduces per-frame jitter). Active only
# while processing video in-memory, or when force_landmark_smoothing is set
# (e.g. for the extract-frames video path). Never applied to unrelated image
# batches. strength in [0,1]; higher = smoother.
landmark_smoothing = True
landmark_smoothing_strength = 0.7
# soft dead-zone as a fraction of face size: keypoint motion below this is treated
# as detector noise and frozen out (kills still-head landmark wobble). ~0.006 of a
# ~400px face is ~2.4px and cuts still-head jitter ~60% while barely affecting real
# motion. 0 disables it. Tune via the "Landmark dead-zone" slider.
landmark_smoothing_deadzone = 0.006
force_landmark_smoothing = False


# High-accuracy 68-point landmarker (FaceFusion 2dfan4, ONNX, MIT). When ON and
# the model file is present, its 68 landmarks replace buffalo_l's landmark_3d_68
# (used for landmark alignment only). 5-point arcface kps and
# landmark_2d_106 are unchanged. Put 2dfan4.onnx in ./models/ or set the path.
use_hi_landmarker = False

# Landmark sanity gate (opt-in). Compares the 5 points derived from the 68
# landmarks against the detector's own 5 kps; if they disagree by more than
# `landmark_sanity_threshold` the 68pt set is treated as broken for that frame
# and the alignment falls back to the kps. OFF by default: the two point sets
# differ systematically, so a low threshold makes the gate fire on ordinary
# frames and the alignment flips basis, which flickers. Only useful if your
# footage has genuine landmark blow-ups at hard poses.
landmark_sanity_gate = False
# Threshold as a FRACTION OF FACE SIZE (the larger side of the face bbox), not
# pixels. 0.10 on a 200px face = 20px of average disagreement. The per-point
# limit is twice this value.
landmark_sanity_threshold = 0.20
hi_landmarker_model_path = ''
hi_landmarker_debug = False

# Faceset (multi-image source) identity averaging. 'robust' (default) drops
# outlier uploads by cosine distance from the group's median embedding, then
# takes a detector-confidence-weighted mean -> identity that resembles the real
# source more closely. 'median' picks the single most-central face. 'mean' is
# the old naive average. outlier_threshold: cosine distance cut for 'robust'
# (lower = stricter; raise toward 1.0 to keep very different angles).
faceset_average_mode = 'robust'
faceset_outlier_threshold = 0.6

# LivePortrait expression restorer (optional, faceswap tab). Re-injects the
# target's real expression onto the swapped face. Default OFF (27 Sep, user;
# heavy model, slower previews on CPU).
expression_restorer = False
expression_restorer_factor = 100    # 0-500 -> blend amount (100 = target amount; user default, was 80)
expression_restore_eyes = True
expression_restore_mouth = True
expression_restore_brows = True
# 0 = no clamp (let the full target expression through). Set ~0.1 to gently
# clamp if very strong expressions ever cause artifacts.
expression_clamp = 0.0
# Amplify the driving expression delta. 1.0 = natural target amount; try 2.0-3.0
# to make subtle expressions clearly visible (may add artifacts if pushed high).
expression_power = 1.0             # multiplier on the delta; 1.0 = target amount
# Preview / frame slider: read new frames from a re-encoded copy of the target
# video with a keyframe every 8 frames (made in the background when the video
# is selected). Renders and the face pickers always read the original.
preview_seek_copy = True
# Keep LivePortrait keypoints 0, 4, 5, 8, 9 from the swapped face (FaceFusion
# never transfers them). A/B toggle for likeness with the restorer on.
expression_keep_structure = False
# Temporal smoothing of the LivePortrait expression vector (video wobble fix).
# Absorbs per-frame noise in the ER expression while still following real
# expression changes. Single control: strength (0 = off). Read live each frame.
expression_smoothing_strength = 0.0
# This generator inverts expression if fed the intuitive way; True applies the
# corrected (verified) mapping so target expression transfers in the right
# direction. Leave True. (Toggle only if a future model export flips again.)
expression_invert_direction = True
# Border feather for the expression restorer: fraction of the crop radius where
# the LivePortrait result fades into the aligned swapped face. Keeps edges clean
# (no 'half face'). 0 disables. 0.2 default.
expression_blend_border = 0.2
# Full LivePortrait pipeline: crop the face the LivePortrait way (wider, incl.
# forehead) from the full frame + use the stitching model to lock pose. This is
# the precise path. Set False to fall back to the in-place arcface-crop method.
expression_full_pipeline = False
expression_stitching = False
lp_crop_size = 512
lp_crop_scale = 2.3
lp_crop_vy = -0.125
# Roll-align the full-pipeline LP crop (upright the face inside it, like arcface).
# Fixes misaligned/ghosted output when the head is strongly tilted (~90 deg).
lp_crop_roll_align = True
# Full-pipeline paste-back calibration: corrects the generator's systematic
# output-framing offset (face slightly shifted/enlarged/rotated after the 2.3x
# crop round-trip). Tune live while watching the preview with full pipeline ON.
# scale 1.0 = no change; dx/dy are fractions of the crop size; rot in degrees.
expression_lp_cal_scale = 1.0
expression_lp_cal_dx = 0.0
expression_lp_cal_dy = 0.0
expression_lp_cal_rot = 0.0
# Adaptive pose lock for the expression restorer (jalur in-place). What gets
# locked is the GLOBAL similarity component of the expression delta; the
# tolerances let genuine global expression through and clamp only the excess:
#   scale_tol   : spread deviation allowed (fraction). 0 = old full scale-lock,
#                 large (e.g. 10) = scale-lock off. 0.04 keeps jaw-drops intact.
#   rot_tol_deg : Kabsch rotation allowed (degrees). 0 = old full rotation-lock,
#                 large (e.g. 180) = rotation-lock off. Logs show real expression
#                 rotation at 0.7-3 deg with junk spikes 4-6.5 deg -> 2.0.
# Translation is always fully locked while expression_pose_lock is True.
# Tune via [expr-delta] (now prints sdev + kabsch vs the tolerances).
expression_pose_lock = True
expression_pose_lock_scale_tol = 0.04
expression_pose_lock_rot_tol = 2.0
# Pose gate: LivePortrait is out-of-distribution at extreme head pose (far back /
# strong profile) and smears the face. Fade the restorer out between 'soft' and
# 'hard' degrees of max(|pitch|,|yaw|), skipping it entirely past 'hard' (keeps
# the clean swapped face). soft/hard in degrees. Set gate False to disable.
expression_pose_gate = False
expression_pose_gate_soft = 45.0
expression_pose_gate_hard = 65.0
# Print per-frame expression diagnostics ([expr-delta]: signal before/after the
# pose lock, % of expression kept, Kabsch rotation angle removed, restorer crop
# resolution). Use this to A/B the pose-lock toggles and expression_power live.
expression_debug = False
# Serialise the LivePortrait sessions across threads (prevents cuDNN/illegal-memory
# crashes when Max Threads is high). True is safe; False = max speed, fewer threads.
expression_serialize = True
# LivePortrait sessions with ONNX Runtime's deterministic compute: the same
# input gives the same expression every time (see Expression_LivePortrait._load).
# False = the kernels cuDNN picks, not repeatable on the GPU.
expression_deterministic = True

no_face_action = 1                  # default: Retry rotated

processing = False

g_current_face_analysis = None
g_current_det_params = None
# Constant module set for face analysis in every mode (see ProcessMgr.initialize).
# Defaulting it here means source-face extraction uses the same analyser as
# processing, so buffalo_l isn't rebuilt on the first swap.
g_desired_face_analysis = ["landmark_3d_68", "landmark_2d_106", "detection", "recognition", "genderage"]


INPUT_FACESETS = []
TARGET_FACES = []



CFG: Settings = None



# --- Diagnostics & onnxruntime CUDA tuning (added for slowness investigation) ---
# When True, prints per-stage timing for each processed/preview frame and logs
# every time a model session is (re)created. This is how we find out whether the
# slowness is a per-frame model reload or slow inference. Set False once diagnosed.
profile_timings = False
# onnxruntime CUDA convolution algo search. The onnxruntime default is
# cuDNN conv algorithm search. ORT default is 'EXHAUSTIVE': it benchmarks conv
# algorithms ONCE per unique shape at the first inference (a one-time startup cost
# of a few seconds), then caches the fastest pick -- ideal for VIDEO where every
# frame is the same size, so steady-state fps is maximal. 'HEURISTIC' skips that
# search (faster startup) but often picks a slower algo -> ~2x slower per frame.
# Quality is identical either way (the chosen algos are mathematically equivalent).
# Use 'HEURISTIC' only if the one-time EXHAUSTIVE search OOMs on a very tight GPU.
cudnn_conv_algo_search = 'EXHAUSTIVE'
# cuDNN conv workspace. False = don't reserve max workspace (saves GBs of GPU RAM)
# BUT forbids cuDNN's fastest conv algos (Winograd/FFT) -> ~2x slower per frame
# across the WHOLE pipeline (swapper+enhancer+mask+LP+detector all read this knob).
# True = fast convs (restores normal speed); zero effect on output quality, it only
# changes which mathematically-equivalent algorithm cuDNN picks.
# On a 24GB GPU (L4) with the full LivePortrait pipeline OFF this fits comfortably.
# If you enable the full LP pipeline (4 nets) + a heavy enhancer and hit CUDA OOM,
# set this back to False to trade speed for memory.
cudnn_conv_use_max_workspace = True

# Preview delivery: the Gradio preview image is sent to the browser (often over a
# slow gradio.live share tunnel) as a full-resolution PNG, which can take a long
# time to appear even though the swap compute is fast. For the PREVIEW ONLY we
# downscale to this max height and let Gradio encode it as JPEG. Final renders are
# unaffected. Raise this if you want a sharper (but slower-to-load) preview.
preview_max_height = 720
