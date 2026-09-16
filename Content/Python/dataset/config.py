"""
Per-project configuration for the underwater dataset-generation pipeline.

EVERYTHING IN THIS FILE IS THE ONLY THING YOU SHOULD NEED TO EDIT to reuse
this pipeline on a new project/level. Every other script imports from here
instead of hardcoding actor names, so a new project mostly means: rename
these constants to match your own scene, then run the scripts in order
(see README block at the bottom of this file).

Written for UE 5.8, tested against /Game/Maps/Dataset_Survey_01 in the
BP_V2 project (a duplicate of BP_UnderWater_Main). See the gotchas at the
bottom before adapting to a new project.
"""

# --- Level / actor identification -------------------------------------------------
SPLINE_ACTOR_LABEL = "SPLINE_SurveyPath_Core"     # actor holding the camera-path SplineComponent

# Actor-label PREFIXES (not exact names) used to find obstacle/rig actors by pattern.
ROCK_LABEL_PREFIX = "SM_URock"                    # every rock mesh actor starts with this
WATER_RIG_LABELS = ("BP_UnderWaterV1", "BP_UnderWater5")  # underwater post-process/fog rig actors;
                                                            # traces must ignore these or every trace
                                                            # just hits the water plane's own collision.
LANDSCAPE_ACTOR_LABEL = "Landscape1"

# --- Clearance / height-policy constants (all in Unreal units = centimeters) ------
CAMERA_CLEARANCE_RADIUS_CM = 80.0     # sphere-trace radius used to check camera-body clearance from rocks
GROUND_CLEARANCE_CM = 150.0           # how far above bare ground/small obstacles the path should float
CLIMB_CAP_CM = 400.0                  # if a rock is taller than this above baseline, DON'T climb over it -
                                       # cap the climb at baseline+CLIMB_CAP_CM instead (keeps path underwater
                                       # near big boulder formations rather than breaching the surface)
GROUND_TOLERANCE_CM = 30.0            # how much shortfall from GROUND_CLEARANCE_CM is tolerated before a
                                       # sample counts as a real "landscape penetration" violation

# --- Capture rig ---------------------------------------------------------------
DATASET_CONTENT_PATH = "/Game/Dataset"      # where render targets / materials live (UE asset path)
RT_RGB_NAME = "RT_Capture_RGB"
RT_DEPTH_NAME = "RT_Capture_Depth"
CAPTURE_RESOLUTION = 1024

# --- Pose sampling ---------------------------------------------------------------
FIXED_CAMERA_PITCH_DEG = -8.0          # constant downward tilt; yaw comes from the spline's own tangent
DEFAULT_SEED = 42

# --- Output (plain disk paths, NOT Unreal asset paths) ---------------------------
# Resolved at runtime via unreal.Paths.project_dir(), see io_paths.py.
DATASET_DISK_SUBDIR = "Dataset"

# ============================================================================
# GOTCHAS FROM THE ORIGINAL PROJECT (read before adapting to a new one)
# ============================================================================
# 1. unreal.SplineComponent.get_location_at_distance_along_spline / rotation_at_distance
#    use an internal arc-length "reparameterization table" that gave measurably wrong
#    positions once the spline had a mix of very short and very long segments (e.g.
#    after inserting corrective points). NEVER trust those distance-based getters for
#    anything precision-sensitive on an edited/irregular spline. Always build your own
#    cumulative-distance array from get_location_at_spline_point() +
#    get_distance_along_spline_at_spline_point(), and do the arc-length interpolation
#    yourself in Python. See path_validation.py / pose_sampling.py for the pattern.
#
# 2. unreal.HitResult properties (location, impact_point, hit_actor, etc.) cannot be
#    read via get_editor_property() in this build ("protected" error) - use
#    hit_result.to_dict() and index into the returned dict instead.
#
# 3. unreal.SceneCaptureComponent2D did NOT evaluate post-process material blendables
#    in this project's editor-world capture context - confirmed with a trivial solid-
#    color test material, tried every SceneCaptureSource variant, render_in_main_renderer,
#    and both component-local AND global (PostProcessVolume) blendables. The SAME
#    material worked fine on the interactive viewport. Root cause not found; if you hit
#    this on a new project, don't burn much time on it - fall back to SCS_SCENE_DEPTH
#    (a raw buffer capture, unaffected) plus screen-space bounding-box projection for
#    occlusion testing instead of a baked-in class-color mask. See capture_rig.py.
#
# 4. Exporting a float-format render target (R32F/RGBA32F, needed for depth in real
#    world units) via unreal.ImageWriteBlueprintLibrary only works with
#    DesiredImageFormat.EXR - PNG/JPG raise "Unsupported texture format".
#
# 5. r.CustomDepth and any render_custom_depth/custom_depth_stencil_value flags set on
#    actors/components are NOT persisted across editor restarts - if you do end up using
#    stencil-based masking, re-run the tagging script every session before capturing.
