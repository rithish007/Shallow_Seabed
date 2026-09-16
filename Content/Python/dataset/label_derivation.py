"""
Step 5 (label derivation) - per-frame YOLO bounding boxes for the 500-frame
survey capture (Dataset/images/{rgb,depth}/frame_#####.*, Dataset/poses_survey_01.json).

DESIGN CHANGE vs. the original plan text: Step 5 was written to read a class-color
mask PNG for per-pixel class matching and box clipping. That mask-material capture
path never worked in this project (SceneCaptureComponent2D silently skips
post-process material blendables - see gotcha #3 in config.py and the Session 2
update in the plan doc), so RGB/mask were replaced with RGB/depth during Step 4.
This module derives labels from CANDIDATE GEOMETRY + the depth buffer instead:

  1. gather_candidates() walks every foliage HISM instance + tagged mesh actor
     ONCE and records its world-space AABB (reuses tag_stencil_classes's
     CLASS_STENCIL_MAP for which meshes/actors belong to which class). This is
     static geometry - cached to disk (candidates_cache.json) so it never needs
     re-running unless the level's foliage/rocks change.
  2. run_labeling() re-plays the saved poses (RGB frames are untouched - already
     on disk from Step 4). For each frame it only re-runs the DEPTH capture (RGB
     capture_source/material is unaffected by any of this), then for every
     candidate within visibility range:
       - projects the world AABB corners to screen space with a manual pinhole
         projection (camera basis from the pose Rotator; FOV/resolution read
         from the live SCS_Capture_RGB/Depth rig - both share fov_angle=90,
         1024x1024, aspect 1:1, confirmed live 2026-07-21),
       - rejects it beyond VISIBILITY_RANGE_CM (fog-blindness cull - Step 3's
         "handled in Step 5" note),
       - rejects boxes under MIN_BOX_PX on a side,
       - samples the depth buffer at 5 points inside the projected box and
         rejects candidates where a majority of samples read a nearer depth
         than the candidate itself (occluded by something else - the
         depth-based replacement for the mask-pixel occlusion test).
  3. Writes YOLO `<class_id> cx cy w h` (normalized, one .txt per frame) into
     Dataset/labels/, plus a dataset.json manifest recording every kept box.

KNOWN LIMITATION vs. the original plan: without the class-color mask, a kept box
is the projected AABB as-is (occlusion-tested by 5 sample points), not "clipped
to the visible extent of class pixels" - a box can include some occluded
interior pixels the 5 samples happened to miss. Acceptable for a feasibility
test; revisit with denser sampling if training shows a real precision problem.

VISIBILITY_RANGE_CM is read LIVE from BP_UnderWater5's 'Fog Distance' BP
variable (the "surviving" rig per the plan's Step 7 DR table) - this is the
actual shader-side visibility cutoff baked into the RGB frames already on disk,
but per config.py's gotchas it is an in-memory, NOT persisted, value. If you
re-run this after an editor restart or a turbidity change, re-check it against
what was live when Step 4's images were captured, or you'll cull to the wrong
range.

Usage:
    import dataset.config as cfg
    import dataset.label_derivation as ld

    candidates = ld.gather_candidates(cfg)            # ~30-90s, one-time
    ld.save_candidates(candidates, ld.candidates_cache_path(cfg))

    candidates = ld.load_candidates(ld.candidates_cache_path(cfg))     # subsequent sessions
    grid = ld.build_grid(candidates)
    vis_range_cm = ld.get_visibility_range_cm()
    poses = ld.pose_sampling.load_manifest(manifest_path)["poses"]

    # chunked, like capture_rig.run_capture - re-render depth + write labels
    ld.run_labeling(cfg, poses, candidates, grid, vis_range_cm,
                    start_frame=0, end_frame=50)
"""
import json
import math
import os

import unreal

import dataset.capture_rig as rig


MIN_BOX_PX = 16
GRID_CELL_CM = 2000.0                  # spatial-hash cell size for the per-frame candidate cull
NEIGHBOR_CELLS = 1                     # search a (2*N+1)^2 block of cells around the camera
DEPTH_OCCLUSION_TOLERANCE_CM = 60.0    # candidate counts as occluded if the sampled depth
                                        # is nearer than (candidate_depth - tolerance)
DEPTH_SAMPLE_POINTS = ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))  # box-relative (u, v)
MIN_VISIBLE_SAMPLE_FRACTION = 0.4      # >=2 of 5 samples must see "this candidate or nearer" to keep the box

CLASS_NAMES = {1: "coral", 2: "kelp", 3: "rock", 4: "sponge"}


def candidates_cache_path(cfg):
    proj_dir = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())
    return os.path.join(proj_dir, cfg.DATASET_DISK_SUBDIR, "candidates_cache.json")


def get_visibility_range_cm(rig_label="BP_UnderWater5"):
    """Reads the live 'Fog Distance' BP variable - see module docstring caveat."""
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = next(a for a in eas.get_all_level_actors() if a.get_actor_label() == rig_label)
    return float(actor.get_editor_property("Fog Distance"))


# ---------------------------------------------------------------------------
# One-time candidate gathering (world-space AABBs), cached to disk
# ---------------------------------------------------------------------------

def _class_of(name, class_stencil_map):
    for sid, prefixes in class_stencil_map.items():
        if any(name.startswith(p) for p in prefixes):
            return sid
    return None


def gather_candidates(cfg):
    """Returns list of {"class_id": int, "center": (x,y,z), "half_extent": (hx,hy,hz)}
    world-space AABBs for every foliage instance / mesh actor matching
    tag_stencil_classes.CLASS_STENCIL_MAP. Static geometry - safe to cache."""
    import dataset.tag_stencil_classes as tag

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = eas.get_all_level_actors()
    candidates = []
    local_box_cache = {}

    for a in actors:
        cls_name = a.get_class().get_name()
        if "InstancedFoliageActor" in cls_name:
            for comp in a.get_components_by_class(unreal.InstancedStaticMeshComponent):
                mesh = comp.get_editor_property("static_mesh")
                if mesh is None:
                    continue
                sid = _class_of(mesh.get_name(), tag.CLASS_STENCIL_MAP)
                if sid is None:
                    continue
                mesh_key = mesh.get_name()
                if mesh_key not in local_box_cache:
                    local_box_cache[mesh_key] = mesh.get_bounding_box()
                local_box = local_box_cache[mesh_key]
                n = comp.get_instance_count()
                for i in range(n):
                    tform = comp.get_instance_transform(i, True)
                    if tform is None:
                        continue
                    world_box = local_box.get_transformed_box(tform)
                    mn, mx = world_box.min, world_box.max
                    candidates.append({
                        "class_id": sid,
                        "center": ((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, (mn.z + mx.z) / 2.0),
                        "half_extent": ((mx.x - mn.x) / 2.0, (mx.y - mn.y) / 2.0, (mx.z - mn.z) / 2.0),
                    })
        else:
            sid = _class_of(a.get_actor_label(), tag.CLASS_STENCIL_MAP)
            if sid is None:
                continue
            origin, extent = a.get_actor_bounds(True)
            candidates.append({
                "class_id": sid,
                "center": (origin.x, origin.y, origin.z),
                "half_extent": (extent.x, extent.y, extent.z),
            })
    return candidates


def save_candidates(candidates, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(candidates, f)


def load_candidates(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Spatial hash grid - 68k+ candidates is too many to project every frame;
# only examine the ones near the camera.
# ---------------------------------------------------------------------------

def build_grid(candidates, cell_size_cm=GRID_CELL_CM):
    grid = {}
    for idx, c in enumerate(candidates):
        cx, cy, _ = c["center"]
        key = (int(cx // cell_size_cm), int(cy // cell_size_cm))
        grid.setdefault(key, []).append(idx)
    return grid


def nearby_candidate_indices(grid, cell_size_cm, cam_x, cam_y, neighbor_cells=NEIGHBOR_CELLS):
    gx, gy = int(cam_x // cell_size_cm), int(cam_y // cell_size_cm)
    idxs = []
    for dx in range(-neighbor_cells, neighbor_cells + 1):
        for dy in range(-neighbor_cells, neighbor_cells + 1):
            idxs.extend(grid.get((gx + dx, gy + dy), []))
    return idxs


# ---------------------------------------------------------------------------
# Manual pinhole projection (world -> screen). Camera basis from the pose's
# Rotator; fov_angle is UE's HORIZONTAL fov - fine to reuse for vertical too
# since CAPTURE_RESOLUTION is square (aspect 1:1, verified live).
# ---------------------------------------------------------------------------

def _project_point(cam_loc, forward, right, up, half_fov_tan, res_x, res_y, wx, wy, wz):
    """Returns (screen_x, screen_y, depth_cm) or None if behind the camera.
    depth_cm is the FORWARD (planar) camera-space depth - same convention as
    the engine's SceneDepth."""
    dx, dy, dz = wx - cam_loc.x, wy - cam_loc.y, wz - cam_loc.z
    x_cam = dx * right.x + dy * right.y + dz * right.z
    y_cam = dx * up.x + dy * up.y + dz * up.z
    z_cam = dx * forward.x + dy * forward.y + dz * forward.z
    if z_cam <= 1.0:
        return None
    sx = (0.5 + (x_cam / z_cam) / (2.0 * half_fov_tan)) * res_x
    sy = (0.5 - (y_cam / z_cam) / (2.0 * half_fov_tan)) * res_y
    return sx, sy, z_cam


def _project_candidate_box(cam_loc, forward, right, up, half_fov_tan, res_x, res_y, candidate):
    cx, cy, cz = candidate["center"]
    hx, hy, hz = candidate["half_extent"]
    xs, ys, depths = [], [], []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                proj = _project_point(cam_loc, forward, right, up, half_fov_tan, res_x, res_y,
                                       cx + sx * hx, cy + sy * hy, cz + sz * hz)
                if proj is None:
                    return None  # a corner is behind/at the camera plane - skip (conservative)
                px, py, pz = proj
                xs.append(px); ys.append(py); depths.append(pz)
    x0, x1 = max(0.0, min(xs)), min(float(res_x), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(res_y), max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1, min(depths)


# ---------------------------------------------------------------------------
# Per-frame labeling
# ---------------------------------------------------------------------------

def run_labeling(cfg, poses, candidates, grid, visibility_range_cm, output_dir,
                  start_frame=0, end_frame=None, world=None):
    """Re-renders ONLY the depth capture at each saved pose (RGB is untouched
    on disk) and writes YOLO label .txt files + returns the per-frame records
    to append into the dataset.json manifest. Mirrors capture_rig.run_capture's
    chunking contract - call in slices of frames per round-trip.
    """
    end_frame = len(poses) if end_frame is None else end_frame
    if world is None:
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = {a.get_actor_label(): a for a in eas.get_all_level_actors()}
    depth_actor = actors["SCS_Capture_Depth"]
    depth_comp = depth_actor.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
    rt_depth = unreal.EditorAssetLibrary.load_asset(f"{cfg.DATASET_CONTENT_PATH}/{cfg.RT_DEPTH_NAME}")

    fov_deg = depth_comp.get_editor_property("fov_angle")
    half_fov_tan = math.tan(math.radians(fov_deg) / 2.0)
    res_x = res_y = cfg.CAPTURE_RESOLUTION

    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    records = []
    for i in range(start_frame, end_frame):
        p = poses[i]
        loc = unreal.Vector(p["location"]["x"], p["location"]["y"], p["location"]["z"])
        rot = unreal.Rotator(p["rotation"]["pitch"], p["rotation"]["yaw"], p["rotation"]["roll"])
        depth_actor.set_actor_location_and_rotation(loc, rot, False, False)
        depth_comp.capture_scene()

        forward, right, up = rot.get_forward_vector(), rot.get_right_vector(), rot.get_up_vector()

        idxs = nearby_candidate_indices(grid, GRID_CELL_CM, loc.x, loc.y)
        boxes = []
        for idx in idxs:
            c = candidates[idx]
            cx, cy, cz = c["center"]
            dist = math.dist((loc.x, loc.y, loc.z), (cx, cy, cz))
            if dist > visibility_range_cm:
                continue
            projected = _project_candidate_box(loc, forward, right, up, half_fov_tan, res_x, res_y, c)
            if projected is None:
                continue
            x0, y0, x1, y1, depth_cm = projected
            if (x1 - x0) < MIN_BOX_PX or (y1 - y0) < MIN_BOX_PX:
                continue

            visible_hits = 0
            for u, v in DEPTH_SAMPLE_POINTS:
                px = int(min(res_x - 1, max(0, x0 + u * (x1 - x0))))
                py = int(min(res_y - 1, max(0, y0 + v * (y1 - y0))))
                sampled_depth = rig.read_depth_pixel(world, rt_depth, px, py)
                if sampled_depth >= depth_cm - DEPTH_OCCLUSION_TOLERANCE_CM:
                    visible_hits += 1
            if visible_hits / len(DEPTH_SAMPLE_POINTS) < MIN_VISIBLE_SAMPLE_FRACTION:
                continue

            cx_n = ((x0 + x1) / 2.0) / res_x
            cy_n = ((y0 + y1) / 2.0) / res_y
            w_n = (x1 - x0) / res_x
            h_n = (y1 - y0) / res_y
            boxes.append({"class_id": c["class_id"], "cx": cx_n, "cy": cy_n, "w": w_n, "h": h_n})

        fnum = p["frame"]
        with open(os.path.join(labels_dir, f"frame_{fnum:05d}.txt"), "w") as f:
            for b in boxes:
                f.write(f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

        records.append({"frame": fnum, "pose": p, "num_boxes": len(boxes), "boxes": boxes})

    return records


def append_manifest(records, manifest_path, class_names=CLASS_NAMES):
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"class_names": class_names, "frames": {}}
    for r in records:
        manifest["frames"][str(r["frame"])] = r
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path
