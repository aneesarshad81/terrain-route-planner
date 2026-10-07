#!/usr/bin/env python3
"""
ridgeway_path_mca_v7_chunkerheuristic.py
Heuristic Chunker-aware MCA -> block-accurate Ridgeway path generator.

Principles:
 - Heuristically search chunk NBT for any plausible 256-length height array.
 - Accept many tag-name variants (bedrock_heights, BedrockHeights, Heights, Heightmap...).
 - If no good array found, fall back to scanning chunk blocks (top -> down).
 - Build world heightmap (Z rows × X cols) using region coordinates, then run A* pathfinder
   at block precision with user-configured leniency / max step.
 - Produce CSV (X,Y,Z), a PNG preview and an optional GeoTIFF for visualization.

Dependencies (install in your virtual environment):
  pip install anvil-parser nbtlib numpy pillow tqdm matplotlib scipy rasterio

Usage example:
  source path/to/your/venv/bin/activate
  python3 ridgeway_path_mca_v7_chunkerheuristic.py \
    --region-dir="path/to/java_1.20.2_world/region" \
    --base-a="0,70,0" \
    --base-b="200,75,300" \
    --out-dir="output" \
    --max-step=6 \
    --leniency=small \
    --snap-radius=96
"""

import argparse
import json
import math
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import sys

# anvil parser & nbtlib
try:
    import anvil
    import nbtlib
except Exception as e:
    print("Missing required libs: install anvil-parser and nbtlib in your venv.")
    raise

# --- Helpers: heuristics to find 256-length height arrays inside NBT --- #
def recursive_find_height_arrays(tag):
    """
    Recursively search an nbtlib tag (or Python structure) for any list/array
    of length 256 that looks like heights (values in plausible Y-range).
    Returns first plausible list or None.
    """
    # accept list-like with length 256
    # tag can be nbtlib's Compound/List/IntArray etc.
    seen = set()
    def is_plausible_list(lst):
        if not hasattr(lst, "__len__"):
            return False
        if len(lst) != 256:
            return False
        # convert to ints and check plausibility
        try:
            vals = [int(x) for x in lst]
        except Exception:
            return False
        # plausible Y-range in MC: typically -64..319 (allow  -128..512 as loose check)
        low, high = min(vals), max(vals)
        if low < -200 or high > 1024:  # garbage check
            return False
        # require not all equal to the void sentinel (-64)
        if all(v == -64 for v in vals):
            return False
        return True

    def walk(node):
        nid = id(node)
        if nid in seen:
            return None
        seen.add(nid)
        # if it's an nbtlib tag with .items(), iterate
        # try list behavior
        try:
            if is_plausible_list(node):
                return [int(x) for x in node]
        except Exception:
            pass
        # If nbtlib tag (Compound), iterate values
        if hasattr(node, "items"):
            for k, v in node.items():
                res = walk(v)
                if res is not None:
                    return res
        # If list-like, iterate children
        if isinstance(node, (list, tuple)):
            for v in node:
                res = walk(v)
                if res is not None:
                    return res
        return None

    return walk(tag)

# --- Read one chunk and try to extract a 16x16 height array (flattened 256) --- #
def extract_chunk_height_from_anvil_chunk(chunk):
    """
    Try multiple heuristics on an anvil.Chunk:
      1) chunk.Heightmaps keys (WORLD_SURFACE / MOTION_BLOCKING / WORLD_SURFACE_WG)
      2) search chunk.nbt recursively for 256-length arrays
      3) fall back to scanning blocks for top non-air
    Returns a 16x16 numpy array of ints (local x across, local z down), or None if failed.
    """
    # 1) direct Heightmaps (common when present)
    try:
        if hasattr(chunk, "Heightmaps") and chunk.Heightmaps:
            # chunk.Heightmaps might be dict-like with keys mapped to arrays
            for key in ("WORLD_SURFACE", "WORLD_SURFACE_WG", "MOTION_BLOCKING", "MOTION_BLOCKING_NO_LEAVES"):
                if key in chunk.Heightmaps:
                    hm = chunk.Heightmaps[key]
                    if len(hm) == 256:
                        arr = np.array(hm, dtype=np.int32).reshape((16,16)).T  # anvil ordering -> (z,x)
                        # sanity: if mostly -64 then reject
                        if np.all(arr == -64):
                            continue
                        return arr
            # otherwise try any 256-length height array
            for k,v in chunk.Heightmaps.items():
                if hasattr(v, "__len__") and len(v)==256:
                    arr = np.array(v, dtype=np.int32).reshape((16,16)).T
                    if np.all(arr == -64):
                        continue
                    return arr
    except Exception:
        pass

    # 2) recursive NBT search
    try:
        # chunk may have attribute 'nbt' (nbtlib tag) or ability to get level tag
        nbt_root = None
        if hasattr(chunk, "nbt") and chunk.nbt is not None:
            nbt_root = chunk.nbt
        else:
            # try to construct a python dict from chunk.__dict__
            try:
                nbt_root = chunk
            except Exception:
                nbt_root = None

        if nbt_root is not None:
            found = recursive_find_height_arrays(nbt_root)
            if found:
                arr = np.array(found, dtype=np.int32).reshape((16,16)).T
                if not np.all(arr == -64):
                    return arr
    except Exception:
        pass

    # 3) fallback: block scan (top-down). This is the slowest but most robust.
    try:
        # anvil.Chunk has get_block(x,y,z) and chunk.sections maybe.
        # We'll scan y from top (319) down to -64 and find first non-air block
        local = np.full((16,16), -64, dtype=np.int32)
        # Get a sensible top Y bound by checking chunk.blocks or sections if available
        topY = 319
        found_any = False
        for lx in range(16):
            for lz in range(16):
                # search down
                for y in range(319, -65, -1):
                    try:
                        blk = chunk.get_block(lx, y, lz)
                    except Exception:
                        blk = None
                    if blk is None:
                        continue
                    # anvil's get_block returns Block object or name; handle both
                    name = None
                    if hasattr(blk, "id") and blk.id is not None:
                        name = str(blk.id)
                    elif hasattr(blk, "name"):
                        name = str(blk.name)
                    elif isinstance(blk, str):
                        name = blk
                    if name is None:
                        continue
                    if "air" not in name.lower():
                        local[lx, lz] = y
                        found_any = True
                        break
        if found_any:
            # transpose to (z,x)
            return local.T
    except Exception:
        pass

    return None

# --- Build full world heightmap from region folder --- #
def build_world_heightmap_from_region_dir(region_dir: Path, verbose=True):
    """
    Reads all r.*.mca files, determines map bounds and origin, then fills an array (z,x)
    with heights using the heuristic extractor above.
    Returns: heightmap (2D np array), origin_x, origin_z
    """
    region_dir = Path(region_dir)
    region_files = sorted([p for p in region_dir.glob("r.*.mca")])
    if not region_files:
        raise RuntimeError("No region files found in region-dir")

    # parse region coords
    def parse_region_name(p: Path):
        stem = p.stem  # r.x.z
        try:
            _, rx, rz = stem.split(".")
            return int(rx), int(rz)
        except Exception:
            raise RuntimeError(f"Bad region filename: {stem}")

    rcoords = [parse_region_name(p) for p in region_files]
    rxs = [r[0] for r in rcoords]
    rzs = [r[1] for r in rcoords]
    rx_min, rx_max = min(rxs), max(rxs)
    rz_min, rz_max = min(rzs), max(rzs)

    # world extents in blocks: each region = 512x512 blocks
    origin_x = rx_min * 512
    origin_z = rz_min * 512
    size_x = (rx_max - rx_min + 1) * 512
    size_z = (rz_max - rz_min + 1) * 512

    if verbose:
        print(f"→ Map bounds: region x [{rx_min}..{rx_max}] z [{rz_min}..{rz_max}]")
        print(f"→ World origin = ({origin_x},{origin_z}) | Map size = ({size_x},{size_z}) blocks")

    # initialize heightmap with sentinel (-32768) to identify missing
    heightmap = np.full((size_z, size_x), -32768, dtype=np.int32)

    # Iterate regions and chunks
    for p in tqdm(region_files, desc="regions"):
        rx, rz = parse_region_name(p)
        # load region
        try:
            reg = anvil.Region.from_file(str(p))
        except Exception as e:
            print(f"Failed to open region {p}: {e}", file=sys.stderr)
            continue

        for cx in range(32):
            for cz in range(32):
                try:
                    chunk = reg.get_chunk(cx, cz)
                except Exception:
                    continue
                # compute world chunk coordinates
                world_chunk_x = rx * 32 + cx
                world_chunk_z = rz * 32 + cz
                # places in block coords
                base_x = (world_chunk_x * 16) - origin_x
                base_z = (world_chunk_z * 16) - origin_z
                # extract 16x16 local heights
                local = extract_chunk_height_from_anvil_chunk(chunk)
                if local is None:
                    continue
                # local is (z=16,x=16) shape -> place into heightmap
                z0, z1 = base_z, base_z + 16
                x0, x1 = base_x, base_x + 16
                # bounds check
                if z0 < 0 or x0 < 0 or z1 > heightmap.shape[0] or x1 > heightmap.shape[1]:
                    # partial overlap: do safe copy
                    for lz in range(16):
                        for lx in range(16):
                            gz = base_z + lz
                            gx = base_x + lx
                            if 0 <= gz < heightmap.shape[0] and 0 <= gx < heightmap.shape[1]:
                                heightmap[gz, gx] = int(local[lz, lx])
                else:
                    heightmap[z0:z1, x0:x1] = local.astype(np.int32)

    # Postprocess: replace any remaining sentinel with -64 (void) or nearest neighbor interpolation
    missing = (heightmap == -32768)
    if missing.any():
        # fill missing with nearest-neighbor interpolation to avoid holes (important for pathfinding)
        from scipy import ndimage
        mask = ~missing
        if mask.sum() == 0:
            # all missing? fail hard
            raise RuntimeError("All heights missing after extraction.")
        # compute distances to nearest known and fill
        coords = np.array(np.nonzero(mask)).T
        # use ndimage distance transform method
        filled = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
        # filled is indices of nearest known for each cell
        zi, xi = filled
        heightmap = heightmap[zi, xi]
    return heightmap, origin_x, origin_z

# --- Simple A* path finder on integer grid (block-accurate, Manhattan+elevation cost) --- #
import heapq
def astar_path_grid(
    heightmap,
    start_px,
    start_pz,
    goal_px,
    goal_pz,
    water_level=68
):
    H, W = heightmap.shape

    def in_bounds(x,z):
        return 0 <= x < W and 0 <= z < H

    neighs = [
        (-1,0),(1,0),
        (0,-1),(0,1),
        (-1,-1),(1,-1),
        (-1,1),(1,1)
    ]

    start = (start_px,start_pz)
    goal  = (goal_px,goal_pz)

    open_set = []
    heapq.heappush(open_set,(abs(goal_px-start_px)+abs(goal_pz-start_pz),0,start))

    came_from = {}
    direction_from = {}
    direction_from[start] = (0, 0)
    gscore = {start:0}

    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            cur = current
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return list(reversed(path))

        cx, cz = current

        for dx, dz in neighs:
            nx = cx + dx
            nz = cz + dz

            if not in_bounds(nx,nz):
                continue

            terrain_y = int(heightmap[nz,nx])

            turn_penalty = 0
            prev_dir = direction_from.get(current, (0, 0))

            if prev_dir != (0, 0):
                if (dx, dz) != prev_dir:

                    if dx == -prev_dir[0] and dz == -prev_dir[1]:
                        turn_penalty = 100

                    elif (
                        abs(dx - prev_dir[0]) <= 1
                        and abs(dz - prev_dir[1]) <= 1
                    ):
                        turn_penalty = 1.5

                    else:
                        turn_penalty = 8

            excavation = max(0, terrain_y - water_level)

            step_cost = 1.0
            step_cost += turn_penalty
            step_cost += excavation * excavation * 0.35

            if excavation > 5:
                step_cost += 50

            if excavation > 15:
                step_cost += 300

            if excavation > 30:
                step_cost += 2500

            if terrain_y <= water_level + 2:
                step_cost *= 0.50
            elif terrain_y <= water_level + 5:
                step_cost *= 0.75

            local_penalty = 0
            for ox in (-1,0,1):
                for oz in (-1,0,1):
                    tx = nx + ox
                    tz = nz + oz
                    if not in_bounds(tx,tz):
                        continue
                    local_penalty += max(0, int(heightmap[tz,tx]) - water_level)

            step_cost += local_penalty * 0.15

            if dx != 0 and dz != 0:
                step_cost += 0.15

            if abs(goal_px - nx) > 50:
                step_cost -= 0.05

            tentative_g = g + step_cost
            neighbor = (nx,nz)

            if neighbor not in gscore or tentative_g < gscore[neighbor]:
                gscore[neighbor] = tentative_g
                heuristic = abs(goal_px-nx) + abs(goal_pz-nz)
                priority = tentative_g + heuristic
                came_from[neighbor] = current
                direction_from[neighbor] = (dx, dz)
                heapq.heappush(open_set,(priority,tentative_g,neighbor))

    return []

# --- Main CLI --- #
def parse_xyz(s):
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Base coords must be x,y,z")
    return float(parts[0]), float(parts[1]), float(parts[2])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region-dir", required=True)
    p.add_argument("--base-a", required=True, type=parse_xyz)
    p.add_argument("--base-b", required=True, type=parse_xyz)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-step", type=int, default=3)
    p.add_argument("--leniency", choices=["strict","small","medium","high"], default="small")
    p.add_argument("--snap-radius", type=int, default=48, help="radius (blocks) to snap base to nearest valid terrain")
    p.add_argument("--water-level", type=int, default=68, help="fixed canal water level")
    args = p.parse_args()

    region_dir = Path(args.region_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build heightmap
    print("🔎 Building heightmap from region files (chunker-heuristic)...")
    heightmap, origin_x, origin_z = build_world_heightmap_from_region_dir(region_dir, verbose=True)
    print(f"→ Heightmap shape (z,x): {heightmap.shape}")
    # Save preview PNG for inspection
    vmin, vmax = int(np.nanmin(heightmap)), int(np.nanmax(heightmap))
    preview = (heightmap - vmin) / max(1, (vmax - vmin))
    preview_img = (np.clip(preview,0,1) * 255).astype(np.uint8)
    im = Image.fromarray(preview_img)
    preview_path = out_dir / "heightmap_from_mca_preview_canal_v2.png"
    im.save(preview_path)
    meta = {"origin": {"x": origin_x, "z": origin_z}, "shape": {"z": int(heightmap.shape[0]), "x": int(heightmap.shape[1])}, "min": int(vmin), "max": int(vmax)}
    with open(out_dir / "heightmap_from_mca_meta_v7.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"🖼️ Preview saved → {preview_path}")
    print(f"🧾 Meta saved → {out_dir / 'heightmap_from_mca_meta_v7.json'}")

    # Snap bases: convert world coords -> pixel indices
    (ax, ay, az) = args.base_a
    (bx, by, bz) = args.base_b
    # world -> pixel: px = int(wx - origin_x), pz = int(wz - origin_z)
    def world_to_pixel(wx, wz):
        px = int(round(wx - origin_x))
        pz = int(round(wz - origin_z))
        return px, pz

    apx, apz = world_to_pixel(ax, az)
    bpx, bpz = world_to_pixel(bx, bz)
    H, W = heightmap.shape
    # If base out-of-bounds or sitting on void sentinel, search within snap-radius for nearest valid cell
    def snap_to_valid(px, pz, radius):
        # if already valid
        if 0<=pz<H and 0<=px<W and heightmap[pz,px] != -32768:
            return px, pz
        best = None
        bestd = None
        for dz in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nx, nz = px+dx, pz+dz
                if nx < 0 or nz < 0 or nx >= W or nz >= H:
                    continue
                v = heightmap[nz, nx]
                if v == -32768:
                    continue
                d = abs(dx) + abs(dz)
                if best is None or d < bestd:
                    best = (nx, nz); bestd = d
        return best

    a_snap = snap_to_valid(apx, apz, args.snap_radius)
    b_snap = snap_to_valid(bpx, bpz, args.snap_radius)
    if a_snap is None or b_snap is None:
        print("❌ Could not snap one or both bases to valid terrain. Try increasing --snap-radius or check region files.")
        sys.exit(1)
    apx, apz = a_snap
    bpx, bpz = b_snap
    ay_real = int(heightmap[apz, apx])
    by_real = int(heightmap[bpz, bpx])
    print(f"📍 Base A snapped → pixel ({apx},{apz}) height ≈ {ay_real}")
    print(f"📍 Base B snapped → pixel ({bpx},{bpz}) height ≈ {by_real}")

    # Run A* pathfinder (block-accurate)
    print(f"🚀 Running pathfinder (max-step={args.max_step}, leniency={args.leniency})...")
    path_pixels = astar_path_grid(heightmap, apx, apz, bpx, bpz, water_level=args.water_level)
    if not path_pixels:
        print("❌ Path not found — try larger --max-step or increase --leniency.")
        sys.exit(1)

    print(f"✅ Path found — steps: {len(path_pixels)}")
    # convert to world coordinates and Y
    path_world = []
    for px, pz in path_pixels:
        wx = origin_x + px
        wz = origin_z + pz
        wy = int(heightmap[pz, px])
        path_world.append((int(wx), int(wy), int(wz)))

    # Save CSV
    csv_path = out_dir / "canal_path_v2.csv"
    with open(csv_path, "w") as fh:
        fh.write("X,Y,Z\n")
        for x,y,z in path_world:
            fh.write(f"{x},{y},{z}\n")
    print(f"💾 Path saved → {csv_path}")

    # Save a labeled PNG visualization (zoomed to bounding box)
    pxs = [p[0]-origin_x for p in [(ax,ay,az),(bx,by,bz)]]
    pzs = [p[2]-origin_z for p in [(ax,ay,az),(bx,by,bz)]]
    # bounding box around path for small preview
    xs = [px for px,pz in path_pixels]
    zs = [pz for px,pz in path_pixels]
    minx, maxx = max(0, min(xs)-16), min(W-1, max(xs)+16)
    minz, maxz = max(0, min(zs)-16), min(H-1, max(zs)+16)
    vis = (heightmap[minz:maxz+1, minx:maxx+1] - vmin) / max(1, (vmax - vmin))
    vis_img = (np.clip(vis,0,1)*255).astype(np.uint8)
    vis_rgb = np.stack([vis_img]*3, axis=2)
    # draw path (red) and bases (green/blue)
    for i,(px,pz) in enumerate(path_pixels):
        vx = px - minx
        vz = pz - minz
        if 0 <= vz < vis_rgb.shape[0] and 0 <= vx < vis_rgb.shape[1]:
            vis_rgb[vz, vx] = [255,0,0]
    # mark bases
    ax_v, az_v = apx - minx, apz - minz
    bx_v, bz_v = bpx - minx, bpz - minz
    def draw_cross(img, cx, cy, col):
        for dx in (-2,-1,0,1,2):
            x = cx+dx
            if 0 <= cy < img.shape[0] and 0 <= x < img.shape[1]:
                img[cy,x] = col
        for dy in (-2,-1,0,1,2):
            y = cy+dy
            if 0 <= y < img.shape[0] and 0 <= cx < img.shape[1]:
                img[y,cx] = col
    draw_cross(vis_rgb, ax_v, az_v, [0,255,0])
    draw_cross(vis_rgb, bx_v, bz_v, [0,0,255])
    out_preview = out_dir / "canal_path_visual_v2.png"
    Image.fromarray(vis_rgb).save(out_preview)
    print(f"🖼️ Visualization saved → {out_preview}")

    print("✅ Done — outputs in:", out_dir)

if __name__ == "__main__":
    main()
