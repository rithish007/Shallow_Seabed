"""Reference example: the exact sequence used to build this project's dataset."""
import os

import unreal

import dataset.config as cfg
import dataset.path_validation as pv
import dataset.pose_sampling as ps
import dataset.capture_rig as rig
import dataset.tag_stencil_classes as tag


def main():
    world = pv.get_world()
    water, rocks, foliage = pv.gather_actors(cfg)
    sp = pv.get_spline_component(cfg.SPLINE_ACTOR_LABEL)

    report = pv.validate(world, sp, water, rocks,
                          sphere_radius_cm=cfg.CAMERA_CLEARANCE_RADIUS_CM,
                          ground_tolerance_cm=cfg.GROUND_TOLERANCE_CM)
    print("BEFORE:", report["summary"])

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

    decim = pv.decimate_safe(world, sp, water, rocks,
                              ground_tolerance_cm=cfg.GROUND_TOLERANCE_CM,
                              sphere_radius_cm=cfg.CAMERA_CLEARANCE_RADIUS_CM)
    print("DECIMATED:", decim)

    poses, total_len = ps.sample_poses(sp, num_frames=500,
                                        fixed_pitch_deg=cfg.FIXED_CAMERA_PITCH_DEG,
                                        seed=cfg.DEFAULT_SEED)
    manifest_path = os.path.join(
        unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()),
        cfg.DATASET_DISK_SUBDIR, "poses_survey_01.json")
    ps.save_manifest(poses, total_len, cfg.DEFAULT_SEED, cfg.FIXED_CAMERA_PITCH_DEG, manifest_path)
    print(f"Saved {len(poses)} poses -> {manifest_path}")

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
