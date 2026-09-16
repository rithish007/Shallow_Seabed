"""
Reference example: the exact sequence used to build this project's dataset.
Not meant to be run blindly end-to-end (path repair especially is worth checking
between steps) - copy/adapt pieces of this into the Unreal Python console or
execute_python_code as needed. Edit config.py first for a new project.

Run with, e.g.:  exec(open(r"<project>/Content/Python/dataset/example_full_run.py").read())
"""
import os

import unreal

import dataset.config as cfg
import dataset.path_validation as pv
import dataset.pose_sampling as ps
import dataset.capture_rig as rig
import dataset.tag_stencil_classes as tag  # optional - see module docstring


def main():
    world = pv.get_world()
    water, rocks, foliage = pv.gather_actors(cfg)
    sp = pv.get_spline_component(cfg.SPLINE_ACTOR_LABEL)

    # 1) Check the path as it currently stands.
    report = pv.validate(world, sp, water, rocks,
                          sphere_radius_cm=cfg.CAMERA_CLEARANCE_RADIUS_CM,
                          ground_tolerance_cm=cfg.GROUND_TOLERANCE_CM)
    print("BEFORE:", report["summary"])

    # 2) If there are real violations, repair in this order: horizontal avoidance
    #    first (rock clearance), THEN ground-penetration fix (it needs to run last
    #    because moving points horizontally changes what's underneath them).
    if report["rock_violations"] > 0:
        pv.repel_from_rocks(world, sp, water, rocks,
                             sphere_radius_cm=cfg.CAMERA_CLEARANCE_RADIUS_CM,
                             ground_clearance_cm=cfg.GROUND_CLEARANCE_CM,
                             climb_cap_cm=cfg.CLIMB_CAP_CM)
    pv.fix_ground_penetration(world, sp, water, rocks,
                               ground_tolerance_cm=cfg.GROUND_TOLERANCE_CM,
                               ground_clearance_cm=cfg.GROUND_CLEARANCE_CM,
                               climb_cap_cm=cfg.CLIMB_CAP_CM)

    report = pv.validate(world, sp, water, rocks,
                          sphere_radius_cm=cfg.CAMERA_CLEARANCE_RADIUS_CM,
                          ground_tolerance_cm=cfg.GROUND_TOLERANCE_CM)
    print("AFTER REPAIR:", report["summary"])

    # 3) Optional: bring the point count down for easier hand-editing. This can
    #    only remove points where BOTH clearances stay within tolerance - if your
    #    terrain/obstacles are dense, the safe minimum may still be higher than
    #    you'd like (loosen ground_tolerance_cm / sphere_radius_cm to trade safety
    #    for fewer points, deliberately, rather than assuming it'll just work).
    decim = pv.decimate_safe(world, sp, water, rocks,
                              ground_tolerance_cm=cfg.GROUND_TOLERANCE_CM,
                              sphere_radius_cm=cfg.CAMERA_CLEARANCE_RADIUS_CM)
    print("DECIMATED:", decim)

    # 4) Sample the FINISHED path (hand-edit it in the editor between step 3 and
    #    here if you want manual control - re-run fix_ground_penetration() +
    #    validate() afterward to catch anything the manual edit reintroduced).
    poses, total_len = ps.sample_poses(sp, num_frames=500,
                                        fixed_pitch_deg=cfg.FIXED_CAMERA_PITCH_DEG,
                                        seed=cfg.DEFAULT_SEED)
    manifest_path = os.path.join(
        unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()),
        cfg.DATASET_DISK_SUBDIR, "poses_survey_01.json")
    ps.save_manifest(poses, total_len, cfg.DEFAULT_SEED, cfg.FIXED_CAMERA_PITCH_DEG, manifest_path)
    print(f"Saved {len(poses)} poses -> {manifest_path}")

    # 5) Build the capture rig and run it (chunked - see capture_rig.run_capture docstring).
    rt_rgb, rt_depth = rig.ensure_render_targets(cfg)
    rgb_actor, depth_actor = rig.ensure_capture_actors(cfg, start_location=unreal.Vector(
        poses[0]["location"]["x"], poses[0]["location"]["y"], poses[0]["location"]["z"]),
        rt_rgb=rt_rgb, rt_depth=rt_depth)

    output_dir = os.path.join(
        unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()),
        cfg.DATASET_DISK_SUBDIR, "images")
    CHUNK = 50
    for start in range(0, len(poses), CHUNK):
        end = min(start + CHUNK, len(poses))
        count, dt = rig.run_capture(poses, rgb_actor, depth_actor, rt_rgb, rt_depth,
                                     output_dir, start_frame=start, end_frame=end)
        print(f"captured [{start},{end}) in {dt:.1f}s")


if __name__ == "__main__":
    main()
