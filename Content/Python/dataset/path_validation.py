"""Validate, repair, and decimate a LINEAR unreal.SplineComponent camera path so it: (a) never dips below the landscape surface, and (b) keeps a minimum clearance from nearby obstacle meshes (e.g. rocks)."""
import math
import unreal


def get_world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def get_all_actors():
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    return eas.get_all_level_actors()


def gather_actors(cfg):
    actors = get_all_actors()
    water = [a for a in actors if a.get_actor_label() in cfg.WATER_RIG_LABELS]
    rocks = [a for a in actors if a.get_actor_label().startswith(cfg.ROCK_LABEL_PREFIX)]
    foliage = [a for a in actors if "Foliage" in a.get_class().get_name()]
    return water, rocks, foliage


def get_spline_actor(label):
    actors = get_all_actors()
    return next(a for a in actors if a.get_actor_label() == label)


def get_spline_component(label):
    return get_spline_actor(label).get_components_by_class(unreal.SplineComponent)[0]


def _trace_down(world, x, y, actors_to_ignore, z_top=3000.0, z_bot=-3000.0):
    top = unreal.Vector(x, y, z_top)
    bot = unreal.Vector(x, y, z_bot)
    hit = unreal.SystemLibrary.line_trace_single(
        world, top, bot, unreal.TraceTypeQuery.ECC_VISIBILITY, False,
        actors_to_ignore, unreal.DrawDebugTrace.NONE, True)
    if hit is None:
        return None
    return hit.to_dict()["location"].z


def required_height(world, x, y, water_actors, rock_actors,
                     ground_clearance_cm=150.0, climb_cap_cm=400.0):
    zg = _trace_down(world, x, y, water_actors)
    if zg is None:
        return None
    zb = _trace_down(world, x, y, water_actors + rock_actors)
    if zb is None:
        return zg + ground_clearance_cm
    if zg - zb > climb_cap_cm:
        return zb + climb_cap_cm
    return zg + ground_clearance_cm


def landscape_height(world, x, y, water_actors, rock_actors, foliage_actors, landscape_label):
    ignore = water_actors + rock_actors + foliage_actors
    top = unreal.Vector(x, y, 5000.0)
    bot = unreal.Vector(x, y, -5000.0)
    hit = unreal.SystemLibrary.line_trace_single(
        world, top, bot, unreal.TraceTypeQuery.ECC_VISIBILITY, False,
        ignore, unreal.DrawDebugTrace.NONE, True)
    if hit is None:
        return None
    d = hit.to_dict()
    actor = d.get("hit_actor")
    if actor is None or actor.get_actor_label() != landscape_label:
        return None
    return d["location"].z


def read_points(spline_component):
    n = spline_component.get_number_of_spline_points()
    pts = [spline_component.get_location_at_spline_point(i, unreal.SplineCoordinateSpace.WORLD) for i in range(n)]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    return xs, ys, zs


def write_points(spline_component, xs, ys, zs):
    pts = [unreal.Vector(xs[i], ys[i], zs[i]) for i in range(len(xs))]
    spline_component.set_spline_points(pts, unreal.SplineCoordinateSpace.WORLD, True)


def validate(world, spline_component, water_actors, rock_actors,
             sphere_radius_cm=80.0, ground_tolerance_cm=30.0, sample_step_cm=50.0):
    xs, ys, zs = read_points(spline_component)
    n = len(xs)
    ground_viol = 0
    n_ground = 0
    worst_ground = 0.0
    rock_viol = 0
    n_rock = 0
    rock_actor_labels = set()

    for i in range(n - 1):
        x0, y0, z0 = xs[i], ys[i], zs[i]
        x1, y1, z1 = xs[i + 1], ys[i + 1], zs[i + 1]
        seg_len = math.dist((x0, y0), (x1, y1))
        n_sub = max(1, int(seg_len / sample_step_cm) + 1)
        px, py, pz = x0, y0, z0
        for k in range(1, n_sub + 1):
            t = k / n_sub
            sx = x0 + (x1 - x0) * t
            sy = y0 + (y1 - y0) * t
            sz = z0 + (z1 - z0) * t

            n_ground += 1
            req = required_height(world, sx, sy, water_actors, rock_actors)
            if req is not None and (req - sz) > ground_tolerance_cm:
                ground_viol += 1
                worst_ground = max(worst_ground, req - sz)

            n_rock += 1
            hit = unreal.SystemLibrary.sphere_trace_single(
                world, unreal.Vector(px, py, pz), unreal.Vector(sx, sy, sz), sphere_radius_cm,
                unreal.TraceTypeQuery.ECC_VISIBILITY, False, water_actors,
                unreal.DrawDebugTrace.NONE, True)
            if hit is not None:
                d = hit.to_dict()
                actor = d.get("hit_actor")
                if actor is not None:
                    rock_viol += 1
                    rock_actor_labels.add(actor.get_actor_label())
            px, py, pz = sx, sy, sz

    summary = (f"ground: {ground_viol}/{n_ground} penetrate >{ground_tolerance_cm}cm "
               f"(worst {worst_ground:.1f}cm) | rock: {rock_viol}/{n_rock} within "
               f"{sphere_radius_cm}cm ({100.0*rock_viol/max(1,n_rock):.1f}%), "
               f"{len(rock_actor_labels)} distinct actors")
    return {
        "ground_violations": ground_viol, "ground_samples": n_ground, "worst_ground_cm": worst_ground,
        "rock_violations": rock_viol, "rock_samples": n_rock, "rock_actors": rock_actor_labels,
        "summary": summary,
    }


def validate_landscape_penetration(world, spline_component, water_actors, rock_actors,
                                    foliage_actors, landscape_label, sample_step_cm=50.0):
    xs, ys, zs = read_points(spline_component)
    n = len(xs)
    violations = 0
    n_samples = 0
    worst = 0.0
    for i in range(n - 1):
        x0, y0, z0 = xs[i], ys[i], zs[i]
        x1, y1, z1 = xs[i + 1], ys[i + 1], zs[i + 1]
        seg_len = math.dist((x0, y0), (x1, y1))
        n_sub = max(1, int(seg_len / sample_step_cm) + 1)
        for k in range(n_sub + 1):
            t = k / n_sub if n_sub > 0 else 0.0
            sx = x0 + (x1 - x0) * t
            sy = y0 + (y1 - y0) * t
            sz = z0 + (z1 - z0) * t
            n_samples += 1
            lz = landscape_height(world, sx, sy, water_actors, rock_actors, foliage_actors, landscape_label)
            if lz is not None and sz < lz:
                violations += 1
                worst = max(worst, lz - sz)
    return {"violations": violations, "samples": n_samples, "worst_cm": worst}


def repel_from_rocks(world, spline_component, water_actors, rock_actors,
                      num_iters=10, sphere_radius_cm=80.0, sample_step_cm=150.0,
                      push_step_cm=150.0, ground_clearance_cm=150.0, climb_cap_cm=400.0,
                      clamp_box=None):
    xs, ys, zs = read_points(spline_component)
    n = len(xs)
    rock_origin_xy = {}
    for r in rock_actors:
        origin, _extent = r.get_actor_bounds(True)
        rock_origin_xy[r.get_actor_label()] = (origin.x, origin.y)

    def cumulative_dists():
        d = [0.0] * n
        for i in range(1, n):
            d[i] = d[i - 1] + math.dist((xs[i - 1], ys[i - 1], zs[i - 1]), (xs[i], ys[i], zs[i]))
        return d

    def bracket(dists, dist):
        lo, hi = 0, n - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if dists[mid] <= dist:
                lo = mid
            else:
                hi = mid
        return lo, hi

    def sample_violations(dists, total, step):
        samples = []
        n_samples = int(total // step) + 1
        for i in range(n_samples + 1):
            d = min(i * step, total)
            lo, hi = bracket(dists, d)
            d0, d1 = dists[lo], dists[hi]
            t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
            samples.append(unreal.Vector(xs[lo] + (xs[hi] - xs[lo]) * t,
                                          ys[lo] + (ys[hi] - ys[lo]) * t,
                                          zs[lo] + (zs[hi] - zs[lo]) * t))
        violations = []
        for i in range(len(samples) - 1):
            hit = unreal.SystemLibrary.sphere_trace_single(
                world, samples[i], samples[i + 1], sphere_radius_cm,
                unreal.TraceTypeQuery.ECC_VISIBILITY, False, water_actors,
                unreal.DrawDebugTrace.NONE, True)
            if hit is None:
                continue
            d = hit.to_dict()
            actor = d.get("hit_actor")
            if actor is not None and actor.get_actor_label() in rock_origin_xy:
                violations.append({"dist": i * step, "rock": actor.get_actor_label()})
        return violations

    for _ in range(num_iters):
        dists = cumulative_dists()
        total = dists[-1]
        violations = sample_violations(dists, total, sample_step_cm)
        if not violations:
            break

        push_x = [0.0] * n
        push_y = [0.0] * n
        for v in violations:
            rx, ry = rock_origin_xy[v["rock"]]
            lo, hi = bracket(dists, v["dist"])
            for idx in (lo, hi):
                dx, dy = xs[idx] - rx, ys[idx] - ry
                mag = math.hypot(dx, dy)
                if mag < 1.0:
                    dx, dy, mag = 1.0, 0.0, 1.0
                push_x[idx] += dx / mag
                push_y[idx] += dy / mag

        moved = set()
        for i in range(n):
            mag = math.hypot(push_x[i], push_y[i])
            if mag > 1e-6:
                xs[i] += (push_x[i] / mag) * push_step_cm
                ys[i] += (push_y[i] / mag) * push_step_cm
                moved.add(i)

        if n > 2:
            nxs, nys = xs[:], ys[:]
            smooth_w = 0.15
            for i in range(1, n - 1):
                nxs[i] = xs[i] * (1 - smooth_w) + (xs[i - 1] + xs[i + 1]) * 0.5 * smooth_w
                nys[i] = ys[i] * (1 - smooth_w) + (ys[i - 1] + ys[i + 1]) * 0.5 * smooth_w
                if abs(nxs[i] - xs[i]) > 0.5 or abs(nys[i] - ys[i]) > 0.5:
                    moved.add(i)
            xs, ys = nxs, nys

        if clamp_box is not None:
            min_x, max_x, min_y, max_y = clamp_box
            for i in range(n):
                xs[i] = max(min_x, min(max_x, xs[i]))
                ys[i] = max(min_y, min(max_y, ys[i]))

        for i in moved:
            z = required_height(world, xs[i], ys[i], water_actors, rock_actors,
                                 ground_clearance_cm, climb_cap_cm)
            if z is not None:
                zs[i] = z

    write_points(spline_component, xs, ys, zs)


def fix_ground_penetration(world, spline_component, water_actors, rock_actors,
                            ground_tolerance_cm=30.0, sample_step_cm=50.0,
                            ground_clearance_cm=150.0, climb_cap_cm=400.0, max_passes=6):
    xs, ys, zs = read_points(spline_component)
    n = len(xs)
    for i in range(n):
        z = required_height(world, xs[i], ys[i], water_actors, rock_actors,
                             ground_clearance_cm, climb_cap_cm)
        if z is not None:
            zs[i] = z

    for _ in range(max_passes):
        new_xs, new_ys, new_zs = [xs[0]], [ys[0]], [zs[0]]
        n_inserted = 0
        for i in range(len(xs) - 1):
            x0, y0, z0 = xs[i], ys[i], zs[i]
            x1, y1, z1 = xs[i + 1], ys[i + 1], zs[i + 1]
            seg_len = math.dist((x0, y0), (x1, y1))
            n_sub = max(1, int(seg_len / sample_step_cm) + 1)
            for k in range(1, n_sub):
                t = k / n_sub
                sx, sy, sz = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t
                req = required_height(world, sx, sy, water_actors, rock_actors,
                                       ground_clearance_cm, climb_cap_cm)
                if req is not None and (req - sz) > ground_tolerance_cm:
                    new_xs.append(sx); new_ys.append(sy); new_zs.append(req)
                    n_inserted += 1
            new_xs.append(x1); new_ys.append(y1); new_zs.append(z1)
        xs, ys, zs = new_xs, new_ys, new_zs
        if n_inserted == 0:
            break

    write_points(spline_component, xs, ys, zs)


def decimate_safe(world, spline_component, water_actors, rock_actors,
                   ground_tolerance_cm=30.0, sphere_radius_cm=80.0, sample_step_cm=50.0):
    import heapq

    xs, ys, zs = read_points(spline_component)
    n0 = len(xs)

    def chord_is_safe(i0, i1):
        x0, y0, z0 = xs[i0], ys[i0], zs[i0]
        x1, y1, z1 = xs[i1], ys[i1], zs[i1]
        seg_len = math.dist((x0, y0), (x1, y1))
        n_sub = max(1, int(seg_len / sample_step_cm) + 1)
        px, py, pz = x0, y0, z0
        for k in range(1, n_sub + 1):
            t = k / n_sub
            sx, sy, sz = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t
            req = required_height(world, sx, sy, water_actors, rock_actors)
            if req is not None and (req - sz) > ground_tolerance_cm:
                return False
            hit = unreal.SystemLibrary.sphere_trace_single(
                world, unreal.Vector(px, py, pz), unreal.Vector(sx, sy, sz), sphere_radius_cm,
                unreal.TraceTypeQuery.ECC_VISIBILITY, False, water_actors,
                unreal.DrawDebugTrace.NONE, True)
            if hit is not None:
                d = hit.to_dict()
                if d.get("hit_actor") is not None:
                    return False
            px, py, pz = sx, sy, sz
        return True

    prev = list(range(-1, n0 - 1))
    nxt = list(range(1, n0 + 1))
    prev[0] = -1
    nxt[n0 - 1] = -1
    alive = [True] * n0
    protected = [False] * n0
    protected[0] = True
    protected[n0 - 1] = True

    def importance(i):
        p, nx = prev[i], nxt[i]
        if p == -1 or nx == -1:
            return float("inf")
        ax, ay = xs[p], ys[p]
        bx, by = xs[i], ys[i]
        cx, cy = xs[nx], ys[nx]
        return abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) * 0.5

    heap = []
    cur_importance = {}
    for i in range(n0):
        if not protected[i]:
            imp = importance(i)
            cur_importance[i] = imp
            heapq.heappush(heap, (imp, i))

    removed = 0
    while heap:
        imp, i = heapq.heappop(heap)
        if not alive[i] or protected[i]:
            continue
        if cur_importance.get(i) != imp:
            continue
        p, nx = prev[i], nxt[i]
        if p == -1 or nx == -1:
            continue
        if chord_is_safe(p, nx):
            alive[i] = False
            nxt[p] = nx
            prev[nx] = p
            removed += 1
            for j in (p, nx):
                if not protected[j]:
                    newimp = importance(j)
                    cur_importance[j] = newimp
                    heapq.heappush(heap, (newimp, j))
        else:
            protected[i] = True

    final_indices = []
    i = 0
    while i != -1:
        final_indices.append(i)
        i = nxt[i]

    new_xs = [xs[i] for i in final_indices]
    new_ys = [ys[i] for i in final_indices]
    new_zs = [zs[i] for i in final_indices]
    write_points(spline_component, new_xs, new_ys, new_zs)
    return {"kept": len(final_indices), "removed": removed, "original": n0}
