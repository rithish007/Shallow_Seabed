"""Sample a finished (hand- or auto-edited) SplineComponent into evenly arc-length- spaced camera poses for dataset capture, and write a reproducible JSON manifest."""
import json
import math
import os
import random

import unreal


def _cumulative_distances(xs, ys, zs):
    n = len(xs)
    d = [0.0] * n
    for i in range(1, n):
        d[i] = d[i - 1] + math.dist((xs[i - 1], ys[i - 1], zs[i - 1]), (xs[i], ys[i], zs[i]))
    return d


def _bracket(dists, dist):
    lo, hi = 0, len(dists) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if dists[mid] <= dist:
            lo = mid
        else:
            hi = mid
    return lo, hi


def sample_poses(spline_component, num_frames, fixed_pitch_deg=-8.0, seed=42):
    random.seed(seed)

    n = spline_component.get_number_of_spline_points()
    pts = [spline_component.get_location_at_spline_point(i, unreal.SplineCoordinateSpace.WORLD) for i in range(n)]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    dists = _cumulative_distances(xs, ys, zs)
    total_len = dists[-1]

    poses = []
    for i in range(num_frames):
        d = (total_len * i) / (num_frames - 1) if num_frames > 1 else 0.0
        lo, hi = _bracket(dists, d)
        d0, d1 = dists[lo], dists[hi]
        t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
        px = xs[lo] + (xs[hi] - xs[lo]) * t
        py = ys[lo] + (ys[hi] - ys[lo]) * t
        pz = zs[lo] + (zs[hi] - zs[lo]) * t
        dx, dy = xs[hi] - xs[lo], ys[hi] - ys[lo]
        yaw = math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0.0
        poses.append({
            "frame": i,
            "distance_along_spline_cm": round(d, 2),
            "location": {"x": round(px, 3), "y": round(py, 3), "z": round(pz, 3)},
            "rotation": {"pitch": fixed_pitch_deg, "yaw": round(yaw, 3), "roll": 0.0},
        })
    return poses, total_len


def save_manifest(poses, total_len_cm, seed, fixed_pitch_deg, out_path):
    manifest = {
        "seed": seed,
        "num_frames": len(poses),
        "total_length_m": round(total_len_cm / 100.0, 2),
        "spacing_m": round(total_len_cm / max(1, len(poses) - 1) / 100.0, 3),
        "fixed_pitch_deg": fixed_pitch_deg,
        "poses": poses,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)
