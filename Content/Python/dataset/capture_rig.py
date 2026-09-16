"""Build (idempotently) a dual SceneCapture2D rig - one plain RGB capture, one raw scene-depth capture - and run it across a pose list to produce a paired image dataset on disk."""
import os
import time

import unreal


def ensure_render_targets(cfg):
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    factory = unreal.TextureRenderTargetFactoryNew()

    def _make(name, fmt):
        path = f"{cfg.DATASET_CONTENT_PATH}/{name}"
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            return unreal.EditorAssetLibrary.load_asset(path)
        rt = asset_tools.create_asset(name, cfg.DATASET_CONTENT_PATH, unreal.TextureRenderTarget2D, factory)
        rt.set_editor_property("size_x", cfg.CAPTURE_RESOLUTION)
        rt.set_editor_property("size_y", cfg.CAPTURE_RESOLUTION)
        rt.set_editor_property("render_target_format", fmt)
        unreal.EditorAssetLibrary.save_asset(path)
        return rt

    rt_rgb = _make(cfg.RT_RGB_NAME, unreal.TextureRenderTargetFormat.RTF_RGBA8)
    rt_depth = _make(cfg.RT_DEPTH_NAME, unreal.TextureRenderTargetFormat.RTF_RGBA32F)
    return rt_rgb, rt_depth


def ensure_capture_actors(cfg, start_location, rt_rgb=None, rt_depth=None):
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = {a.get_actor_label(): a for a in eas.get_all_level_actors()}

    if rt_rgb is None or rt_depth is None:
        rt_rgb, rt_depth = ensure_render_targets(cfg)

    def _spawn(label):
        if label in actors:
            return actors[label]
        a = eas.spawn_actor_from_class(unreal.SceneCapture2D, start_location, unreal.Rotator(0, 0, 0))
        a.set_actor_label(label)
        return a

    rgb_actor = _spawn("SCS_Capture_RGB")
    depth_actor = _spawn("SCS_Capture_Depth")

    rgb_comp = rgb_actor.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
    rgb_comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    rgb_comp.set_editor_property("texture_target", rt_rgb)
    rgb_comp.capture_every_frame = False

    depth_comp = depth_actor.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
    depth_comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_DEPTH)
    depth_comp.set_editor_property("texture_target", rt_depth)
    depth_comp.capture_every_frame = False

    return rgb_actor, depth_actor


def _image_write_options(fmt):
    opts = unreal.ImageWriteOptions()
    opts.set_editor_property("async_", False)
    opts.set_editor_property("overwrite_file", True)
    opts.set_editor_property("format", fmt)
    return opts


def run_capture(poses, rgb_actor, depth_actor, rt_rgb, rt_depth, output_dir,
                 chunk=None, start_frame=0, end_frame=None):
    end_frame = len(poses) if end_frame is None else end_frame
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    rgb_comp = rgb_actor.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
    depth_comp = depth_actor.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
    opts_png = _image_write_options(unreal.DesiredImageFormat.PNG)
    opts_exr = _image_write_options(unreal.DesiredImageFormat.EXR)

    t0 = time.time()
    count = 0
    for i in range(start_frame, end_frame):
        p = poses[i]
        loc = unreal.Vector(p["location"]["x"], p["location"]["y"], p["location"]["z"])
        rot = unreal.Rotator(p["rotation"]["pitch"], p["rotation"]["yaw"], p["rotation"]["roll"])
        rgb_actor.set_actor_location_and_rotation(loc, rot, False, False)
        depth_actor.set_actor_location_and_rotation(loc, rot, False, False)
        rgb_comp.capture_scene()
        depth_comp.capture_scene()

        fnum = p["frame"]
        unreal.ImageWriteBlueprintLibrary.export_to_disk(
            rt_rgb, os.path.join(rgb_dir, f"frame_{fnum:05d}.png"), opts_png)
        unreal.ImageWriteBlueprintLibrary.export_to_disk(
            rt_depth, os.path.join(depth_dir, f"frame_{fnum:05d}.exr"), opts_exr)
        count += 1

    return count, time.time() - t0


def read_depth_pixel(world, rt_depth, x, y):
    color = unreal.RenderingLibrary.read_render_target_raw_pixel(world, rt_depth, x, y, False)
    return color.r
