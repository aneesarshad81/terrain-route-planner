#!/usr/bin/env python3
# ridgeway_path_mca_v7_simplified_csv.py
# v7 logic with simplified CSV straight-segment compression

import argparse, json, math, sys, os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import anvil
import nbtlib

# ----------------------------------------------------------
#  RECURSIVE HEIGHT ARRAY FINDER (FROM V7)
# ----------------------------------------------------------
def recursive_find_height_arrays(tag):
    seen = set()

    def plausible(lst):
        try:
            if len(lst) != 256:
                return False
            vals = [int(x) for x in lst]
            low, high = min(vals), max(vals)
            if low < -200 or high > 1024:
                return False
            if all(v == -64 for v in vals):
                return False
            return True
        except:
            return False

    def walk(node):
        nid = id(node)
        if nid in seen: return None
        seen.add(nid)

        # direct list check
        if hasattr(node, "__len__"):
            try:
                if plausible(node):
                    return [int(x) for x in node]
            except:
                pass

        # dict/compound
        if hasattr(node, "items"):
            for k,v in node.items():
                r = walk(v)
                if r is not None:
                    return r

        # lists
        if isinstance(node, (list,tuple)):
            for v in node:
                r = walk(v)
                if r is not None:
                    return r
        return None

    return walk(tag)

# ----------------------------------------------------------
#  CHUNK HEIGHT EXTRACTION
# ----------------------------------------------------------
def extract_chunk_height(chunk):
    # 1) Try Heightmaps first
    try:
        if hasattr(chunk, "Heightmaps"):
            for key in ("WORLD_SURFACE","MOTION_BLOCKING","WORLD_SURFACE_WG"):
                if key in chunk.Heightmaps:
                    hm = chunk.Heightmaps[key]
                    if len(hm) == 256:
                        arr = np.array(hm, dtype=np.int32).reshape((16,16)).T
                        if not np.all(arr == -64):
                            return arr
    except:
        pass

    # 2) Recursive find
    try:
        root = chunk.nbt if hasattr(chunk, "nbt") else chunk
        found = recursive_find_height_arrays(root)
        if found:
            arr = np.array(found, dtype=np.int32).reshape((16,16)).T
            if not np.all(arr == -64):
                return arr
    except:
        pass

    # 3) Full block scan
    local = np.full((16,16), -64, dtype=np.int32)
    found_any = False
    for lx in range(16):
        for lz in range(16):
            for y in range(319, -65, -1):
                try:
                    blk = chunk.get_block(lx,y,lz)
                except:
                    blk=None
                if blk is None:
                    continue
                name = getattr(blk,"id",None) or getattr(blk,"name",None) or (blk if isinstance(blk,str) else None)
                if name and "air" not in str(name).lower():
                    local[lx,lz] = y
                    found_any = True
                    break
    if found_any:
        return local.T
    return None

# ----------------------------------------------------------
#  BUILD FULL HEIGHTMAP (v7 logic)
# ----------------------------------------------------------
def build_heightmap(region_dir, verbose=True):
    region_dir = Path(region_dir)
    files = sorted(region_dir.glob("r.*.mca"))
    if not files:
        raise RuntimeError("No .mca files found")

    def parse_name(p):
        _,rx,rz = p.stem.split(".")
        return int(rx), int(rz)

    rcoords = [parse_name(p) for p in files]
    rx_min = min(r[0] for r in rcoords)
    rz_min = min(r[1] for r in rcoords)
    rx_max = max(r[0] for r in rcoords)
    rz_max = max(r[1] for r in rcoords)

    origin_x = rx_min*512
    origin_z = rz_min*512
    size_x = (rx_max-rx_min+1)*512
    size_z = (rz_max-rz_min+1)*512

    if verbose:
        print(f"→ Map bounds: region x [{rx_min}..{rx_max}] z [{rz_min}..{rz_max}]")
        print(f"→ World origin = ({origin_x},{origin_z}) | Map size = ({size_x},{size_z}) blocks")

    hmap = np.full((size_z,size_x), -32768, dtype=np.int32)

    for p in tqdm(files, desc="regions"):
        rx,rz = parse_name(p)
        try:
            reg = anvil.Region.from_file(str(p))
        except Exception as e:
            print("Bad region:",p,e)
            continue

        for cx in range(32):
            for cz in range(32):
                try:
                    ch = reg.get_chunk(cx,cz)
                except:
                    continue
                wx_chunk = rx*32 + cx
                wz_chunk = rz*32 + cz

                base_x = (wx_chunk*16) - origin_x
                base_z = (wz_chunk*16) - origin_z

                arr = extract_chunk_height(ch)
                if arr is None:
                    continue

                for lz in range(16):
                    for lx in range(16):
                        gx = base_x + lx
                        gz = base_z + lz
                        if 0 <= gx < size_x and 0 <= gz < size_z:
                            hmap[gz,gx] = int(arr[lz,lx])

    # fill missing
    miss = (hmap == -32768)
    if miss.any():
        from scipy import ndimage
        mask = ~miss
        if mask.sum()==0:
            raise RuntimeError("All heights missing")
        filled = ndimage.distance_transform_edt(miss, return_distances=False, return_indices=True)
        zi,xi = filled
        hmap = hmap[zi,xi]

    return hmap, origin_x, origin_z

# ----------------------------------------------------------
# A* v7 (unchanged from your working version)
# ----------------------------------------------------------
import heapq
def astar(hmap, sx,sz, gx,gz, max_step, leniency):
    H,W = hmap.shape
    def ok(x,z): return 0<=x<W and 0<=z<H
    mul = {"strict":1,"small":1.5,"medium":2.5,"high":4}.get(leniency,1.5)
    step_lim = int(math.ceil(max_step*mul))

    neighs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]

    start=(sx,sz)
    goal=(gx,gz)
    open=[]
    heapq.heappush(open,(0,0,start))
    gscore={start:0}
    came={}

    while open:
        f,g,cur = heapq.heappop(open)
        if cur==goal:
            path=[]
            c=cur
            while c in came:
                path.append(c)
                c=came[c]
            path.append(start)
            return list(reversed(path))

        cx,cz = cur
        ch = int(hmap[cz,cx])
        for dx,dz in neighs:
            nx,nz = cx+dx, cz+dz
            if not ok(nx,nz): continue
            nh=int(hmap[nz,nx])
            if abs(nh-ch) > step_lim: continue

            cost = 1 + abs(nh-ch)*0.5 + (1.4 if dx!=0 and dz!=0 else 1)
            ng = g + cost
            if (nx,nz) not in gscore or ng < gscore[(nx,nz)]:
                gscore[(nx,nz)] = ng
                came[(nx,nz)]=cur
                pri = ng + (abs(gx-nx)+abs(gz-nz))
                heapq.heappush(open,(pri,ng,(nx,nz)))
    return []

# ----------------------------------------------------------
# Simplify path into straight segments
# ----------------------------------------------------------
def simplify(path, hmap, origin_x, origin_z):
    """
    path = [(px,pz)...]
    Returns a list of (wx,wy,wz) simplified
    """
    if len(path)<=2:
        return [(origin_x+px, int(hmap[pz,px]), origin_z+pz) for px,pz in path]

    simp=[]
    simp.append(path[0])

    # direction tracking
    def dir(a,b): return (b[0]-a[0], b[1]-a[1])

    prev_dir = dir(path[0], path[1])

    for i in range(1, len(path)-1):
        d = dir(path[i], path[i+1])
        if d != prev_dir:
            simp.append(path[i])
        prev_dir = d

    simp.append(path[-1])

    # convert to world coords
    out=[]
    for px,pz in simp:
        wx = origin_x+px
        wz = origin_z+pz
        wy = int(hmap[pz,px])
        out.append((wx,wy,wz))
    return out

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def parse_xyz(s):
    x,y,z = s.split(",")
    return float(x),float(y),float(z)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--region-dir", required=True)
    p.add_argument("--base-a", required=True, type=parse_xyz)
    p.add_argument("--base-b", required=True, type=parse_xyz)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-step", type=int, default=6)
    p.add_argument("--leniency", default="small")
    p.add_argument("--snap-radius", type=int, default=96)
    args=p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("🔎 Building heightmap (v7 simplified)...")
    hmap, origin_x, origin_z = build_heightmap(args.region_dir, verbose=True)

    # world→pixel
    def wp(wx,wz): return int(wx-origin_x), int(wz-origin_z)

    apx,apz = wp(args.base_a[0], args.base_a[2])
    bpx,bpz = wp(args.base_b[0], args.base_b[2])

    H,W = hmap.shape

    # snap bases
    def snap(px,pz):
        if 0<=pz<H and 0<=px<W and hmap[pz,px]!=-32768:
            return px,pz
        best=None
        bestd=None
        for dz in range(-args.snap_radius, args.snap_radius+1):
            for dx in range(-args.snap_radius, args.snap_radius+1):
                nx, nz = px+dx, pz+dz
                if 0<=nx<W and 0<=nz<H and hmap[nz,nx]!=-32768:
                    d=abs(dx)+abs(dz)
                    if best is None or d<bestd:
                        best=(nx,nz); bestd=d
        return best

    a_snap = snap(apx,apz)
    b_snap = snap(bpx,bpz)
    if a_snap is None or b_snap is None:
        print("❌ Snap failed")
        sys.exit(1)

    apx,apz = a_snap
    bpx,bpz = b_snap
    print("📍 Base A snapped:", a_snap, "Base B:", b_snap)

    # run A*
    print("🚀 Pathfinding...")
    path = astar(hmap, apx,apz, bpx,bpz, args.max_step, args.leniency)
    if not path:
        print("❌ Path not found")
        sys.exit(1)

    print("✅ Path found:", len(path))

    # simplify
    spath = simplify(path, hmap, origin_x, origin_z)

    # output
    csv_path = out/"ridgeway_path_block_v7_simplified.csv"
    with open(csv_path,"w") as f:
        f.write("X,Y,Z\n")
        for x,y,z in spath:
            f.write(f"{x},{y},{z}\n")

    print("💾 Simplified CSV saved →", csv_path)


if __name__ == "__main__":
    main()
