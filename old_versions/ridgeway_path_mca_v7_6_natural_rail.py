#!/usr/bin/env python3
"""
ridgeway_path_mca_v7_6_natural_rail.py
Ridgeway Path Generator v7.6 — always rail-optimized

Rail principles:
 - Uses anvil-parser (or anvil-parser2) + nbtlib to parse chunks at a semantic level.
 - Heuristically searches parsed NBT for ANY plausible 256-length height array (handles chunker/extra tags).
 - If no array found, falls back to scanning actual blocks (get_block) top->down (robust but slower).
 - Builds a full world heightmap (z,x), smooths it strongly, then runs an A* pathfinder tuned for rails:

Rail tuning (baked in):
 - 4-connected moves only (no diagonals) → cleaner rail curves
 - Default max-step = 1 block (vertical) → gentle slopes
 - Strong smoothing (default smooth-window=25) → avoids twitchy paths
 - Higher slope penalty (default slope-weight=0.16) → prefers flatter terrain
 - No leniency scaling: allowed vertical step = max-step directly
"""

import argparse
import json
import math
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import heapq

# numeric / smoothing
from scipy import ndimage

# Try to import anvil-parser (or anvil-parser2)
anvil = None
try:
    import anvil  # common package name for both parser variants
    anvil_found = True
except Exception:
    anvil_found = False
    try:
        # some forks expose as anvil_parser or anvil_parser2; try common fallbacks
        import anvil_parser as anvil  # fallback
        anvil_found = True
    except Exception:
        try:
            import anvil_parser2 as anvil
            anvil_found = True
        except Exception:
            anvil_found = False

if not anvil_found:
    print("ERROR: anvil-parser (or anvil-parser2) not found in venv. Install one of them:")
    print("  pip install anvil-parser")
    print("  OR")
    print("  pip install git+https://github.com/0xTiger/anvil-parser2")
    sys.exit(1)

# nbtlib is used for recursive scanning
try:
    import nbtlib
except Exception:
    print("ERROR: nbtlib not found (pip install nbtlib).")
    sys.exit(1)


# ---------------------------
# Helper: recursive heuristic to find 256-length arrays
# ---------------------------
def recursive_find_height_arrays(node, seen=None):
    """
    Recursively search a parsed NBT (nbtlib Compound/List or plain python structure)
    for a list/array of length 256 that looks like column heights. Return first plausible list or None.
    Heuristics:
      - length == 256
      - values convertible to int
      - values within a plausible range (we allow broad -200..2000) but reject all-void arrays (-64)
    """
    if seen is None:
        seen = set()

    nid = id(node)
    if nid in seen:
        return None
    seen.add(nid)

    # If it's an nbtlib tag or other sequence-like with len()
    try:
        if hasattr(node, "__len__") and not isinstance(node, (str, bytes)):
            try:
                if len(node) == 256:
                    vals = [int(x) for x in node]
                    if min(vals) < -200 or max(vals) > 2000:
                        return None
                    if all(v == -64 for v in vals):
                        return None
                    return vals
            except Exception:
                pass
    except Exception:
        pass

    # mapping-like
    if hasattr(node, "items"):
        for _, v in node.items():
            res = recursive_find_height_arrays(v, seen)
            if res is not None:
                return res

    # iterable/list-like
    if isinstance(node, (list, tuple)):
        for v in node:
            res = recursive_find_height_arrays(v, seen)
            if res is not None:
                return res

    # tag with .value
    if hasattr(node, "value"):
        try:
            res = recursive_find_height_arrays(node.value, seen)
            if res is not None:
                return res
        except Exception:
            pass

    return None


# ---------------------------
# Extract heights for a single chunk (16x16) using heuristic order
# ---------------------------
def extract_chunk_height_from_chunk(chunk):
    """
    Input: chunk object returned by anvil.Region.get_chunk(cx,cz) from parser.
    Output: 16x16 numpy array with heights (z rows, x cols) or None.
    """
    # 1) direct Heightmaps
    try:
        if hasattr(chunk, "Heightmaps") and chunk.Heightmaps:
            hm_dict = chunk.Heightmaps
            for key in ("WORLD_SURFACE", "WORLD_SURFACE_WG", "MOTION_BLOCKING", "MOTION_BLOCKING_NO_LEAVES"):
                if key in hm_dict:
                    raw = hm_dict[key]
                    if hasattr(raw, "__len__") and len(raw) == 256:
                        arr = np.array(raw, dtype=np.int32).reshape((16, 16)).T
                        if not np.all(arr == -64):
                            return arr
            # fallback: any plausible 256 array
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

    # 2) recursive NBT search
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

    # 3) fallback to top-down block scanning
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

                    name = None
                    if blk is None:
                        continue
                    if isinstance(blk, str):
                        name = blk
                    else:
                        if hasattr(blk, "id") and blk.id is not None:
                            name = str(blk.id)
                        elif hasattr(blk, "name"):
                            name = str(blk.name)
                        elif hasattr(blk, "state") and isinstance(blk.state, str):
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


# ---------------------------
# Build full world heightmap from region folder
# ---------------------------
def build_world_heightmap_from_region_dir(region_dir: Path, verbose=True):
    """
    Read r.x.z.mca files, compute world origin and size, iterate regions/chunks and
    extract heights with heuristics above.
    Returns (heightmap np.array (z,x), origin_x, origin_z).
    """
    region_dir = Path(region_dir)
    region_files = sorted(region_dir.glob("r.*.mca"))
    if not region_files:
        raise RuntimeError("No region files found in region-dir")

    def parse_region_name(p: Path):
        st = p.stem  # r.x.z
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


# ---------------------------
# Pathfinder (A* on integer grid) — rail style only
# ---------------------------
def astar_path_grid(heightmap, start_px, start_pz, goal_px, goal_pz, max_step=1, slope_weight=0.16):
    """
    Rail-optimized pathfinder:
      - 4-connected grid (N/E/S/W only)
      - allowed vertical step = max_step (no leniency scaling)
      - strong penalty for vertical change via slope_weight
    """
    H, W = heightmap.shape

    def in_bounds(x, z):
        return 0 <= x < W and 0 <= z < H

    allowed_step = max(1, int(max_step))
    neighs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # no diagonals

    start = (start_px, start_pz)
    goal = (goal_px, goal_pz)
    open_heap = []
    start_h = abs(goal_px - start_px) + abs(goal_pz - start_pz)
    heapq.heappush(open_heap, (start_h, 0.0, start))
    came_from = {}
    gscore = {start: 0.0}
    visited = set()

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current == goal:
            rev = []
            cur = current
            while cur in came_from:
                rev.append(cur)
                cur = came_from[cur]
            rev.append(start)
            return list(reversed(rev))

        if current in visited:
            continue
        visited.add(current)

        cx, cz = current
        ch = int(heightmap[cz, cx])

        for dx, dz in neighs:
            nx, nz = cx + dx, cz + dz
            if not in_bounds(nx, nz):
                continue
            nh = int(heightmap[nz, nx])

            if abs(nh - ch) > allowed_step:
                continue

            base_cost = 1.0  # all moves = 1 block horizontal
            slope_cost = slope_weight * abs(nh - ch)
            tentative_g = g + base_cost + slope_cost
            neigh = (nx, nz)

            if neigh not in gscore or tentative_g < gscore[neigh]:
                gscore[neigh] = tentative_g
                # Manhattan heuristic
                priority = tentative_g + (abs(goal_px - nx) + abs(goal_pz - nz))
                came_from[neigh] = current
                heapq.heappush(open_heap, (priority, tentative_g, neigh))

    return []


# ---------------------------
# Utility: gaussian smoothing of heightmap
# ---------------------------
def smooth_heightmap(heightmap, smooth_window):
    sigma = max(0.5, float(smooth_window) / 6.0)
    sm = ndimage.gaussian_filter(heightmap.astype(np.float32), sigma=sigma, truncate=3.0)
    sm_rounded = np.rint(sm).astype(np.int32)
    return sm_rounded


# ---------------------------
# CLI / main runner
# ---------------------------
def parse_xyz(s):
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Base coords must be x,y,z")
    return float(parts[0]), float(parts[1]), float(parts[2])


def main():
    p = argparse.ArgumentParser(
        description="Ridgeway path generator v7.6 (MCA direct, always rail-optimized)"
    )
    p.add_argument("--region-dir", required=True, help="Folder containing r.*.mca files")
    p.add_argument("--base-a", required=True, type=parse_xyz, help="Base A (x,y,z)")
    p.add_argument("--base-b", required=True, type=parse_xyz, help="Base B (x,y,z)")
    p.add_argument("--out-dir", required=True, help="Output folder")
    p.add_argument(
        "--max-step",
        type=int,
        default=1,
        help="Max allowed single-step vertical diff (blocks) [rail default = 1]",
    )
    p.add_argument(
        "--snap-radius",
        type=int,
        default=96,
        help="Snap radius (blocks) to find nearest valid terrain for bases",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Gaussian smoothing window (rail default = 25)",
    )
    p.add_argument(
        "--slope-weight",
        type=float,
        default=0.16,
        help="Slope penalty weight (rail default = 0.16)",
    )
    args = p.parse_args()

    region_dir = Path(args.region-dir) if False else Path(args.region_dir)  # keep lints happy
    region_dir = Path(args.region_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔎 Building heightmap from region files (this may take time)...")
    heightmap, origin_x, origin_z = build_world_heightmap_from_region_dir(region_dir, verbose=True)
    H, W = heightmap.shape
    vmin, vmax = int(np.min(heightmap)), int(np.max(heightmap))

    # Save meta + preview
    preview = (heightmap - vmin) / max(1, (vmax - vmin))
    preview_img = (np.clip(preview, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(preview_img).save(out_dir / "heightmap_from_mca_preview_v7_6.png")
    meta = {
        "origin": {"x": int(origin_x), "z": int(origin_z)},
        "shape": {"z": int(H), "x": int(W)},
        "min": int(vmin),
        "max": int(vmax),
    }
    with open(out_dir / "heightmap_from_mca_meta_v7_6.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"🖼️ Preview saved → {out_dir / 'heightmap_from_mca_preview_v7_6.png'}")
    print(f"🧾 Meta saved → {out_dir / 'heightmap_from_mca_meta_v7_6.json'}")

    # Smoothing (rail defaults are strong)
    heightmap_used = heightmap.copy()
    if args.smooth_window and args.smooth_window > 1:
        print(f"🧩 Applying Gaussian smoothing (window={args.smooth_window}) for rail-friendly terrain...")
        heightmap_used = smooth_heightmap(heightmap, args.smooth_window)

    # Snap bases
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
        print("❌ Could not snap one or both bases to valid terrain. Increase --snap-radius or check region files.")
        sys.exit(1)

    apx, apz = a_snap
    bpx, bpz = b_snap
    ay_real = int(heightmap[apz, apx])
    by_real = int(heightmap[bpz, bpx])
    print(f"📍 Base A snapped → pixel ({apx},{apz}) height ≈ {ay_real}")
    print(f"📍 Base B snapped → pixel ({bpx},{bpz}) height ≈ {by_real}")

    # A* pathfinder (rail-only)
    print(
        f"🚀 Running rail A* pathfinder (max-step={args.max_step}, "
        f"slope_weight={args.slope_weight}, smooth_window={args.smooth_window})..."
    )
    path_pixels = astar_path_grid(
        heightmap_used,
        apx,
        apz,
        bpx,
        bpz,
        max_step=args.max_step,
        slope_weight=args.slope_weight,
    )
    if not path_pixels:
        print("❌ Path not found — try larger --snap-radius, a slightly larger --max-step, or smaller --slope-weight.")
        sys.exit(1)
    print(f"✅ Path found — steps: {len(path_pixels)}")

    # Convert pixel path to world coords
    path_world = []
    for px, pz in path_pixels:
        wx = origin_x + px
        wz = origin_z + pz
        wy = int(heightmap[pz, px])
        path_world.append((int(wx), int(wy), int(wz)))

    # Save main CSV
    csv_path = out_dir / "ridgeway_path_block_v7_6.csv"
    with open(csv_path, "w") as fh:
        fh.write("X,Y,Z\n")
        for x, y, z in path_world:
            fh.write(f"{x},{y},{z}\n")
    print(f"💾 Path saved → {csv_path}")

    # Save changes CSV
    changes_path = out_dir / "ridgeway_path_changes_v7_6.csv"
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
                f"{idx-1},{prev[0]},{prev[1]},{prev[2]},{curr[0]},{curr[1]},{curr[2]},{dx},{dy},{dz},{changed_str}\n"
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

    # draw path (red)
    for (px, pz) in path_pixels:
        vx = px - minx
        vz = pz - minz
        if 0 <= vz < vis_rgb.shape[0] and 0 <= vx < vis_rgb.shape[1]:
            vis_rgb[vz, vx] = [255, 0, 0]

    # mark bases
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

    draw_cross(vis_rgb, ax_v, az_v, [0, 255, 0])
    draw_cross(vis_rgb, bx_v, bz_v, [0, 0, 255])
    vis_out = out_dir / "ridgeway_path_visual_v7_6.png"
    Image.fromarray(vis_rgb).save(vis_out)
    print(f"🖼️ Visualization saved → {vis_out}")

    print("✅ Done — outputs in:", out_dir)
    print(f"→ Start height (Base A): {ay_real} | End height (Base B): {by_real}")
    print("Notes: Rail-optimized: 4-direction moves only, max-step & slope-weight tuned for rails by default.")


if __name__ == "__main__":
    main()
