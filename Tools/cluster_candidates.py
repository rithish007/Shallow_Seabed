"""
Quick, cheap PREVIEW of a fundamentally different fix: 3D-proximity clustering
of the raw candidate geometry (Dataset/candidates_cache.json - the 68,808
foliage-instance/rock AABBs Step 5 gathered), BEFORE any per-frame projection.

Why this exists: the 2D IoU-merge in merge_labels.py only fixes boxes that
already overlap heavily in a given frame's screen space. It can't turn a
"carpet" of 50 separate-but-adjacent coral instances into one blob, because
none of them overlap enough in 2D. To actually get toward "a handful of boxes
per distinct colony," the merge has to happen in 3D, once, on the real world
geometry - two candidates of the same class are unioned if the gap between
their bounding spheres is <= --radius (i.e. touching or within radius cm of
each other), using union-find over a spatial grid so it stays fast at 68k
candidates.

This script ONLY reports cluster counts (no Unreal, no frame projection) so
you can sanity-check a radius choice in seconds before committing to a full
(slow) re-run of the per-frame Unreal depth-capture + projection pipeline.

Usage:
    python Tools/cluster_candidates.py --radius 150
    python Tools/cluster_candidates.py --radius 150,300,600   # compare several
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset")
CLASS_NAMES = {1: "coral", 2: "kelp", 3: "rock", 4: "sponge"}


def load_candidates():
    with open(os.path.join(DATASET_DIR, "candidates_cache.json"), "r") as f:
        return json.load(f)


def cluster(candidates, radius_cm):
    """Union-find over a spatial grid (cell = radius_cm, per class), merging
    candidates whose bounding-sphere gap <= radius_cm. Returns list of merged
    clusters: {"class_id", "center": (x,y,z), "half_extent": (hx,hy,hz), "merged_from": int}."""
    n = len(candidates)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    sphere_r = []
    grid = defaultdict(list)
    cell = radius_cm if radius_cm > 0 else 1.0
    for idx, c in enumerate(candidates):
        hx, hy, hz = c["half_extent"]
        r = math.sqrt(hx * hx + hy * hy + hz * hz)
        sphere_r.append(r)
        cx, cy, _ = c["center"]
        key = (c["class_id"], int(cx // cell), int(cy // cell))
        grid[key].append(idx)

    for idx, c in enumerate(candidates):
        cx, cy, cz = c["center"]
        cls = c["class_id"]
        gx, gy = int(cx // cell), int(cy // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cls, gx + dx, gy + dy), []):
                    if j <= idx:
                        continue
                    other = candidates[j]
                    ox, oy, oz = other["center"]
                    dist = math.dist((cx, cy, cz), (ox, oy, oz))
                    gap = dist - sphere_r[idx] - sphere_r[j]
                    if gap <= radius_cm:
                        union(idx, j)

    groups = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)

    clusters = []
    for members in groups.values():
        xs0, ys0, zs0, xs1, ys1, zs1 = [], [], [], [], [], []
        for idx in members:
            c = candidates[idx]
            cx, cy, cz = c["center"]
            hx, hy, hz = c["half_extent"]
            xs0.append(cx - hx); ys0.append(cy - hy); zs0.append(cz - hz)
            xs1.append(cx + hx); ys1.append(cy + hy); zs1.append(cz + hz)
        x0, y0, z0 = min(xs0), min(ys0), min(zs0)
        x1, y1, z1 = max(xs1), max(ys1), max(zs1)
        clusters.append({
            "class_id": candidates[members[0]]["class_id"],
            "center": ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
            "half_extent": ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2),
            "merged_from": len(members),
        })
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=str, default="150", help="comma-separated radii (cm) to try")
    args = ap.parse_args()

    candidates = load_candidates()
    print(f"loaded {len(candidates)} raw candidates")
    by_class_raw = Counter(c["class_id"] for c in candidates)
    print("raw per-class:", {CLASS_NAMES[k]: v for k, v in by_class_raw.items()})
    print()

    for radius_str in args.radius.split(","):
        radius = float(radius_str)
        clusters = cluster(candidates, radius)
        by_class = Counter(c["class_id"] for c in clusters)
        sizes = sorted((c["merged_from"] for c in clusters), reverse=True)
        print(f"--- radius={radius}cm -> {len(clusters)} clusters "
              f"(from {len(candidates)}, {100*(1-len(clusters)/len(candidates)):.1f}% reduction) ---")
        print("  per-class:", {CLASS_NAMES[k]: v for k, v in by_class.items()})
        print("  largest clusters (instance count):", sizes[:10])
        print()


if __name__ == "__main__":
    main()
