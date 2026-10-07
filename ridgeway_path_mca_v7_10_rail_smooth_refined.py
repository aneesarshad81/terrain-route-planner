#!/usr/bin/env python3
"""
ridgeway_path_mca_v7_10_rail_smooth_refined.py

Ridgeway Rail Path Generator v7.10 — "Natural smooth, fewer wiggles"

- Based on the v7.8 rail-strict engine (not v7.9).
- Uses MCA region files (anvil / anvil-parser2) to build a heightmap.
- Optionally smooths terrain (Gaussian) for rail-friendly contours.
- Rail-strict A*:
    * 4-direction grid (N/E/S/W).
    * abs(dy) <= max_step (default 1 for rails).
    * No sloped corners (turns only between flat steps).
    * Slopes are straight segments only.
- NEW in v7.10:
    * "Anti-wiggle" penalty:
        - Small extra cost when you turn again too soon
          (run length < min_straight_before_turn).
        - Does NOT penalize big, natural curves.
        - No global preference for a big L-shaped path.
- Same heightmap is used for BOTH search and output Y-values.
- Built-in rail legality check at the end:
    * Aborts (no CSV) if any illegal step is found.
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import heapq

from scipy import ndimage

# Try to import anvil-parser (or anvil-parser2)
anvil = None
try:
    import anvil
    anvil_found = True
except Exception:
    anvil_found = False
    try:
        import anvil_parser as anvil
        anvil_found = True
    except Exception:
        try:
            import anvil_parser2 as anvil
            anvil_found = True
        except Exception:
            anvil_found = False

if not anvil_found:
    print("ERROR: anvil-parser (or anvil-parser2) not found in venv.")
    print("Install one of:")
    print("  pip install anvil-parser")
    print("  or")
    print("  pip install git+https://github.com/0xTiger/anvil-parser2")
    sys.exit(1)

# nbtlib for recursive scanning
try:
    import nbtlib
except Exception:
    print("ERROR: nbtlib not found (pip install nbtlib).")
    sys.exit(1)


# ---------------------------------------------------------------------
# Recursive 256-length height array finder
# ---------------------------------------------------------------------
def recursive_find_height_arrays(node, seen=None):
    if seen is None:
        seen = set()
    nid = id(node)
    if nid in seen:
        return None
    seen.add(nid)

    try:
        if hasattr(node, "__len__") and not isinstance(node, (str, bytes)):
            try:
                if len(node) == 256:
                    vals = [int(x) for x in node]
                    low, high = min(vals), max(vals)
                    if low < -200 or high > 2000:
                        return None
                    if all(v == -64 for v in vals):
                        return None
                    return vals
            except Exception:
                pass
    except Exception:
        pass

    if hasattr(node, "items"):
        for _, v in node.items():
            res = recursive_find_height_arrays(v, seen)
            if res is not None:
                return res

    if isinstance(node, (list, tuple)):
        for v in node:
            res = recursive_find_height_arrays(v, seen)
            if res is not None:
                return res

    if hasattr(node, "value"):
        try:
            res = recursive_find_height_arrays(node.value, seen)
            if res is not None:
                return res
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------
# Chunk height extraction (16x16)
# ---------------------------------------------------------------------
def extract_chunk_height_from_chunk(chunk):
    # 1) Heightmaps
    try:
        if hasattr(chunk, "Heightmaps") and chunk.Heightmaps:
            hm_dict = chunk.Heightmaps
            for key in ("WORLD_SURFACE", "WORLD_SURFACE_WG",
                        "MOTION_BLOCKING", "MOTION_BLOCKING_NO_LEAVES"):
                if key in hm_dict:
                    raw = hm_dict[key]
                    if hasattr(raw, "__len__") and len(raw) == 256:
                        arr = np.array(raw, dtype=np.int32).reshape((16, 16)).T
                        if not np.all(arr == -64):
                            return arr
            for v in hm_dict.values():
                try:
                    if hasattr(v, "__len__") and len(v) == 256:
                        arr = np.array(v, dtype=np.int32).reshape((16, 16)).T
                        if not np.all(arr == -64):
                            return arr
                except Exception:
                    continue
    except Exception:
        pass

    # 2) Recursive NBT
    try:
        nbt_root = None
        if hasattr(chunk, "nbt") and chunk.nbt is not None:
            nbt_root = chunk.nbt
        else:
            try:
                nbt_root = chunk.__dict__
            except Exception:
                nbt_root = None
        if nbt_root is not None:
            found = recursive_find_height_arrays(nbt_root)
            if found:
                arr = np.array(found, dtype=np.int32).reshape((16, 16)).T
                if not np.all(arr == -64):
                    return arr
    except Exception:
        pass

    # 3) Fallback: top-down block scan
    try:
        local = np.full((16, 16), -64, dtype=np.int32)
        found_any = False
        for lx in range(16):
            for lz in range(16):
                y_found = -64
                for y in range(319, -65, -1):
                    try:
                        if hasattr(chunk, "get_block"):
                            blk = chunk.get_block(lx, y, lz)
                        elif hasattr(chunk, "get_block_at"):
                            blk = chunk.get_block_at(lx, y, lz)
                        else:
                            blk = None
                    except Exception:
                        blk = None
                    if blk is None:
                        continue
                    name = None
                    if isinstance(blk, str):
                        name = blk
                    else:
                        if getattr(blk, "id", None) is not None:
                            name = str(blk.id)
                        elif getattr(blk, "name", None) is not None:
                            name = str(blk.name)
                        elif getattr(blk, "state", None) and isinstance(blk.state, str):
                            name = blk.state
                    if name is None:
                        continue
                    if "air" not in name.lower():
                        y_found = y
                        found_any = True
                        break
                local[lx, lz] = y_found
        if found_any:
            return local.T
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------
# World heightmap from region dir
# ---------------------------------------------------------------------
def build_world_heightmap_from_region_dir(region_dir: Path, verbose=True):
    region_dir = Path(region_dir)
    region_files = sorted(region_dir.glob("r.*.mca"))
    if not region_files:
        raise RuntimeError("No region files found in region-dir")

    def parse_region_name(p: Path):
        st = p.stem
        parts = st.split(".")
        if len(parts) != 3:
            raise RuntimeError(f"Bad region filename: {st}")
        return int(parts[1]), int(parts[2])

    coords = [parse_region_name(p) for p in region_files]
    rxs = [c[0] for c in coords]
    rzs = [c[1] for c in coords]
    rx_min, rx_max = min(rxs), max(rxs)
    rz_min, rz_max = min(rzs), max(rzs)

    origin_x = rx_min * 512
    origin_z = rz_min * 512
    size_x = (rx_max - rx_min + 1) * 512
    size_z = (rz_max - rz_min + 1) * 512

    if verbose:
        print(f"→ Map bounds: region x [{rx_min}..{rx_max}] z [{rz_min}..{rz_max}]")
        print(f"→ World origin = ({origin_x},{origin_z}) | Map size = ({size_x},{size_z}) blocks")

    MISSING = -32768
    heightmap = np.full((size_z, size_x), MISSING, dtype=np.int32)

    for p in tqdm(region_files, desc="regions"):
        rx, rz = parse_region_name(p)
        try:
            reg = anvil.Region.from_file(str(p))
        except Exception as e:
            try:
                reg = anvil.Region(str(p))
            except Exception as e2:
                print(f"Failed to open region {p}: {e} / {e2}", file=sys.stderr)
                continue

        for cx in range(32):
            for cz in range(32):
                try:
                    chunk = reg.get_chunk(cx, cz)
                except Exception:
                    continue
                world_chunk_x = rx * 32 + cx
                world_chunk_z = rz * 32 + cz
                base_x = (world_chunk_x * 16) - origin_x
                base_z = (world_chunk_z * 16) - origin_z
                local = extract_chunk_height_from_chunk(chunk)
                if local is None:
                    continue
                for lz in range(16):
                    for lx in range(16):
                        gx = base_x + lx
                        gz = base_z + lz
                        if 0 <= gz < heightmap.shape[0] and 0 <= gx < heightmap.shape[1]:
                            heightmap[gz, gx] = int(local[lz, lx])

    missing_mask = (heightmap == MISSING)
    if missing_mask.all():
        raise RuntimeError("All heights missing after extraction.")
    if missing_mask.any():
        print("→ Filling missing cells by nearest-neighbor interpolation...")
        filled = ndimage.distance_transform_edt(
            missing_mask, return_distances=False, return_indices=True
        )
        zi, xi = filled
        heightmap = heightmap[zi, xi]

    return heightmap, origin_x, origin_z


# ---------------------------------------------------------------------
# Rail-strict A* with anti-wiggle turn penalty (local only)
# ---------------------------------------------------------------------
def astar_path_grid_rail_smooth_refined(
    heightmap_used,
    start_px,
    start_pz,
    goal_px,
    goal_pz,
    max_step=1,
    slope_weight=0.16,
    short_turn_penalty=0.8,
    min_straight_before_turn=3,
):
    """
    Rail-legal A* with local "anti-wiggle" behavior.

    - 4-neighbor grid (N/E/S/W).
    - abs(dy) <= max_step per step.
    - No sloped corners.
    - Turn penalty ONLY when we try to turn again
      before we've gone min_straight_before_turn blocks
      in the previous direction.
    - No global "fewer turns is always better" bias.
    """
    H, W = heightmap_used.shape

    def in_bounds(x, z):
        return 0 <= x < W and 0 <= z < H

    allowed_step = max(1, int(max_step))

    DIRS = [
        (0, -1),  # 0: N
        (1, 0),   # 1: E
        (0, 1),   # 2: S
        (-1, 0),  # 3: W
    ]

    # state: (x, z, dir_idx, last_slope, run_len)
    start_state = (start_px, start_pz, -1, False, 0)

    open_heap = []
    start_h = abs(goal_px - start_px) + abs(goal_pz - start_pz)
    heapq.heappush(open_heap, (start_h, 0.0, start_state))

    gscore = {start_state: 0.0}
    came_from = {}
    visited = set()

    while open_heap:
        f, g, state = heapq.heappop(open_heap)
        x, z, dir_idx, last_slope, run_len = state

        if (x, z) == (goal_px, goal_pz):
            path_states = []
            cur = state
            while cur in came_from:
                path_states.append(cur)
                cur = came_from[cur]
            path_states.append(start_state)
            path_states.reverse()
            return [(sx, sz) for (sx, sz, _, _, _) in path_states]

        if state in visited:
            continue
        visited.add(state)

        ch = int(heightmap_used[z, x])

        for new_dir_idx, (dx, dz) in enumerate(DIRS):
            nx = x + dx
            nz = z + dz
            if not in_bounds(nx, nz):
                continue

            nh = int(heightmap_used[nz, nx])
            dy = nh - ch
            if abs(dy) > allowed_step:
                continue
            new_slope = (dy != 0)

            # sloped-corner rule
            if dir_idx != -1 and new_dir_idx != dir_idx:
                if last_slope or new_slope:
                    continue

            # run length update
            if dir_idx == -1:
                new_run_len = 1
            elif new_dir_idx == dir_idx:
                new_run_len = run_len + 1
            else:
                # changed direction (flat to flat because of rule above)
                new_run_len = 1

            base_cost = 1.0
            slope_cost = slope_weight * abs(dy)

            # local anti-wiggle: if we turn before min_straight_before_turn
            # in the previous direction, add a penalty.
            turn_cost = 0.0
            if dir_idx != -1 and new_dir_idx != dir_idx:
                if run_len < min_straight_before_turn:
                    turn_cost = short_turn_penalty

            tentative_g = g + base_cost + slope_cost + turn_cost

            new_state = (nx, nz, new_dir_idx, new_slope, new_run_len)
            if new_state not in gscore or tentative_g < gscore[new_state]:
                gscore[new_state] = tentative_g
                h = abs(goal_px - nx) + abs(goal_pz - nz)
                priority = tentative_g + h
                came_from[new_state] = state
                heapq.heappush(open_heap, (priority, tentative_g, new_state))

    return []


# ---------------------------------------------------------------------
# Rail legality checker
# ---------------------------------------------------------------------
def check_rail_legality(path_world, max_step):
    illegal = []
    prev_dir = None
    prev_dy = 0

    for i in range(len(path_world) - 1):
        x1, y1, z1 = path_world[i]
        x2, y2, z2 = path_world[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        if   (dx, dz) == (0, -1): d = 0
        elif (dx, dz) == (1, 0):  d = 1
        elif (dx, dz) == (0, 1):  d = 2
        elif (dx, dz) == (-1, 0): d = 3
        else:
            illegal.append((i, "non-cardinal", dx, dy, dz))
            prev_dir, prev_dy = None, dy
            continue

        if abs(dy) > max_step:
            illegal.append((i, "too_steep", dx, dy, dz))

        if prev_dir is not None and d != prev_dir:
            if prev_dy != 0 or dy != 0:
                illegal.append((i, "sloped_corner", dx, dy, dz))

        prev_dir, prev_dy = d, dy

    return illegal


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------
def parse_xyz(s):
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Base coords must be x,y,z")
    return float(parts[0]), float(parts[1]), float(parts[2])


def main():
    p = argparse.ArgumentParser(
        description="Ridgeway rail path generator v7.10 (natural smooth, fewer wiggles)"
    )
    p.add_argument("--region-dir", required=True, help="Folder containing r.*.mca files")
    p.add_argument("--base-a", required=True, type=parse_xyz, help="Base A (x,y,z)")
    p.add_argument("--base-b", required=True, type=parse_xyz, help="Base B (x,y,z)")
    p.add_argument("--out-dir", required=True, help="Output folder")
    p.add_argument("--max-step", type=int, default=1,
                   help="Max allowed single-step vertical diff (blocks). Rails = 1.")
    p.add_argument("--snap-radius", type=int, default=96,
                   help="Snap radius (blocks) to find nearest valid terrain for bases")
    p.add_argument("--smooth-window", type=int, default=25,
                   help="Gaussian smoothing window (rail default = 25)")
    p.add_argument("--slope-weight", type=float, default=0.16,
                   help="Slope penalty weight (rail default = 0.16)")
    p.add_argument("--short-turn-penalty", type=float, default=0.8,
                   help="Extra cost when turning before a short straight (anti-wiggle).")
    p.add_argument("--min-straight-before-turn", type=int, default=3,
                   help="Minimum straight length before turns are 'free' of extra penalty.")
    args = p.parse_args()

    region_dir = Path(args.region_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔎 Building heightmap from region files (this may take time)...")
    heightmap, origin_x, origin_z = build_world_heightmap_from_region_dir(region_dir, verbose=True)
    H, W = heightmap.shape
    vmin, vmax = int(np.min(heightmap)), int(np.max(heightmap))

    # Save preview + meta
    preview = (heightmap - vmin) / max(1, (vmax - vmin))
    preview_img = (np.clip(preview, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(preview_img).save(out_dir / "heightmap_from_mca_preview_v7_10.png")
    meta = {
        "origin": {"x": int(origin_x), "z": int(origin_z)},
        "shape": {"z": int(H), "x": int(W)},
        "min": int(vmin),
        "max": int(vmax),
    }
    with open(out_dir / "heightmap_from_mca_meta_v7_10.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"🖼️ Preview saved → {out_dir / 'heightmap_from_mca_preview_v7_10.png'}")
    print(f"🧾 Meta saved → {out_dir / 'heightmap_from_mca_meta_v7_10.json'}")

    # Smoothing (for A* and final Y values)
    if args.smooth_window and args.smooth_window > 1:
        print(f"🧩 Applying Gaussian smoothing (window={args.smooth_window}) for rail-friendly terrain...")
        sm = ndimage.gaussian_filter(
            heightmap.astype(np.float32),
            sigma=max(0.5, float(args.smooth_window) / 6.0),
            truncate=3.0,
        )
        heightmap_used = np.rint(sm).astype(np.int32)
    else:
        heightmap_used = heightmap.copy()

    # Snap bases using original heightmap
    (ax, ay, az) = args.base_a
    (bx, by, bz) = args.base_b

    def world_to_pixel(wx, wz):
        px = int(round(wx - origin_x))
        pz = int(round(wz - origin_z))
        return px, pz

    apx, apz = world_to_pixel(ax, az)
    bpx, bpz = world_to_pixel(bx, bz)

    def snap_to_valid(px, pz, radius):
        if 0 <= pz < H and 0 <= px < W and heightmap[pz, px] != -32768:
            return px, pz
        best = None
        bestd = 10**9
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, nz = px + dx, pz + dz
                if nx < 0 or nz < 0 or nx >= W or nz >= H:
                    continue
                v = heightmap[nz, nx]
                if v == -32768:
                    continue
                d = abs(dx) + abs(dz)
                if d < bestd:
                    best = (nx, nz)
                    bestd = d
        return best

    a_snap = snap_to_valid(apx, apz, args.snap_radius)
    b_snap = snap_to_valid(bpx, bpz, args.snap_radius)
    if a_snap is None or b_snap is None:
        print("❌ Could not snap one or both bases to valid terrain.")
        sys.exit(1)

    apx, apz = a_snap
    bpx, bpz = b_snap
    ay_real = int(heightmap[apz, apx])
    by_real = int(heightmap[bpz, bpx])
    print(f"📍 Base A snapped → pixel ({apx},{apz}) height ≈ {ay_real}")
    print(f"📍 Base B snapped → pixel ({bpx},{bpz}) height ≈ {by_real}")

    print(
        f"🚀 Running rail-strict A* v7.10 (max-step={args.max_step}, "
        f"slope_weight={args.slope_weight}, smooth_window={args.smooth_window}, "
        f"short_turn_penalty={args.short_turn_penalty}, "
        f"min_straight_before_turn={args.min_straight_before_turn})..."
    )
    path_pixels = astar_path_grid_rail_smooth_refined(
        heightmap_used,
        apx,
        apz,
        bpx,
        bpz,
        max_step=args.max_step,
        slope_weight=args.slope_weight,
        short_turn_penalty=args.short_turn_penalty,
        min_straight_before_turn=args.min_straight_before_turn,
    )
    if not path_pixels:
        print("❌ Path not found — try adjusting parameters or snap radius.")
        sys.exit(1)
    print(f"✅ Path found — steps: {len(path_pixels)}")

    # World coordinates from heightmap_used
    path_world = []
    for px, pz in path_pixels:
        wx = origin_x + px
        wz = origin_z + pz
        wy = int(heightmap_used[pz, px])
        path_world.append((int(wx), int(wy), int(wz)))

    # Rail legality check
    illegal = check_rail_legality(path_world, max_step=args.max_step)
    if illegal:
        print("❌ Rail legality check FAILED. Example illegal steps:")
        for item in illegal[:20]:
            idx, kind, dx, dy, dz = item
            print(f"  idx={idx}, type={kind}, delta=({dx},{dy},{dz})")
        print(f"(Total illegal steps: {len(illegal)})")
        print("Aborting without writing CSV.")
        sys.exit(1)
    else:
        print("✅ Rail legality check passed: no non-cardinal, too-steep, or sloped-corner steps.")

    # Save CSV
    csv_path = out_dir / "ridgeway_path_block_v7_10.csv"
    with open(csv_path, "w") as fh:
        fh.write("X,Y,Z\n")
        for x, y, z in path_world:
            fh.write(f"{x},{y},{z}\n")
    print(f"💾 Path saved → {csv_path}")

    # Changes CSV
    changes_path = out_dir / "ridgeway_path_changes_v7_10.csv"
    with open(changes_path, "w") as fh:
        fh.write("idx,prevX,prevY,prevZ,currX,currY,currZ,dx,dy,dz,changed\n")
        prev = path_world[0]
        for idx, curr in enumerate(path_world[1:], start=1):
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            dz = curr[2] - prev[2]
            changed = []
            if dx != 0:
                changed.append("X")
            if dy != 0:
                changed.append("Y")
            if dz != 0:
                changed.append("Z")
            changed_str = "|".join(changed) if changed else "none"
            fh.write(
                f"{idx-1},{prev[0]},{prev[1]},{prev[2]},"
                f"{curr[0]},{curr[1]},{curr[2]},"
                f"{dx},{dy},{dz},{changed_str}\n"
            )
            prev = curr
    print(f"🧾 Changes saved → {changes_path}")

    # Visualization
    xs = [px for px, pz in path_pixels]
    zs = [pz for px, pz in path_pixels]
    minx, maxx = max(0, min(xs) - 24), min(W - 1, max(xs) + 24)
    minz, maxz = max(0, min(zs) - 24), min(H - 1, max(zs) + 24)
    vis = (heightmap[minz:maxz + 1, minx:maxx + 1] - vmin) / max(1, (vmax - vmin))
    vis_img = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
    vis_rgb = np.stack([vis_img] * 3, axis=2)

    for (px, pz) in path_pixels:
        vx = px - minx
        vz = pz - minz
        if 0 <= vz < vis_rgb.shape[0] and 0 <= vx < vis_rgb.shape[1]:
            vis_rgb[vz, vx] = [255, 0, 0]

    ax_v, az_v = apx - minx, apz - minz
    bx_v, bz_v = bpx - minx, bpz - minz

    def draw_cross(img, cx, cy, col):
        for dx in (-3, -2, -1, 0, 1, 2, 3):
            x = cx + dx
            if 0 <= cy < img.shape[0] and 0 <= x < img.shape[1]:
                img[cy, x] = col
        for dy in (-3, -2, -1, 0, 1, 2, 3):
            y = cy + dy
            if 0 <= y < img.shape[0] and 0 <= cx < img.shape[1]:
                img[y, cx] = col

    draw_cross(vis_rgb, ax_v, az_v, [0, 255, 0])  # A = green
    draw_cross(vis_rgb, bx_v, bz_v, [0, 0, 255])  # B = blue

    vis_out = out_dir / "ridgeway_path_visual_v7_10.png"
    Image.fromarray(vis_rgb).save(vis_out)
    print(f"🖼️ Visualization saved → {vis_out}")

    print("✅ Done — outputs in:", out_dir)
    print(f"→ Start height (Base A): {ay_real} | End height (Base B): {by_real}")
    print("Notes: v7.10 rail-strict A* with local anti-wiggle penalty; natural, smooth, "
          "fewer micro turns; CSV Y uses the same smoothed heightmap as the search.")


if __name__ == "__main__":
    main()
