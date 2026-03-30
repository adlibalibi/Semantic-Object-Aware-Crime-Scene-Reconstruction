"""
Object-Aware Scene Reconstruction Pipeline
Based on Structured3D Dataset

Files used:
  - annotation_3d.json : Room geometry (junctions, planes, lines, semantics)
  - bbox_3d.json       : Object bounding boxes (OBBs) in 3D
  - instance.png       : Instance segmentation mask (panorama or perspective)

This script:
  1. Parses and visualises the 3D room layout (walls/floors/ceilings)
  2. Loads all object OBBs and places them in the scene
  3. Renders a top-down 2D floor plan with objects
  4. Renders an interactive 3D view (Open3D) — optional on CPU
  5. Lays the groundwork for the visibility / occlusion validation step

Requirements (install once):
    pip install open3d numpy matplotlib pillow

For Colab GPU:
    !pip install open3d-python   # or build from source; use headless mode
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – works everywhere
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import os

# ──────────────────────────────────────────────
# PATHS  (edit these to match your local layout)
# ──────────────────────────────────────────────
BASE = r"D:\College Materials\Sem 8\CV\Project\Data"
SCENE_ID = "scene_00001"

ANNOTATION_3D = os.path.join(
    BASE, "Structured3D_annotation_3d", "Structured3D", SCENE_ID, "annotation_3d.json"
)
BBOX_3D = os.path.join(
    BASE, "Structured3D_bbox", "Structured3D", SCENE_ID, "bbox_3d.json"
)
INSTANCE_PANORAMA = os.path.join(
    BASE, "Structured3D_bbox", "Structured3D", SCENE_ID,
    "2D_rendering", "906322", "panorama", "full", "instance.png"
)
INSTANCE_PERSPECTIVE = os.path.join(
    BASE, "Structured3D_bbox", "Structured3D", SCENE_ID,
    "2D_rendering", "906322", "perspective", "full", "0", "instance.png"
)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════

def load_annotation(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def load_bboxes(path: str) -> list:
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════
# 2.  OBB UTILITIES
#     Each bbox has:
#       centroid : [cx, cy, cz]   (mm in Structured3D)
#       coeffs   : [hx, hy, hz]   half-extents
#       basis    : 3×3 rotation   (row = local axis)
# ══════════════════════════════════════════════

def obb_corners(box: dict) -> np.ndarray:
    """Return the 8 corners of an oriented bounding box."""
    c  = np.array(box["centroid"])           # (3,)
    h  = np.array(box["coeffs"])             # half-extents (3,)
    R  = np.array(box["basis"])              # (3,3)  rows = local axes

    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ])  # (8,3)

    # Each corner = centroid + sum_i (sign_i * h_i * R_i)
    corners = c + (signs * h) @ R            # (8,3)
    return corners


def obb_footprint_xy(box: dict) -> np.ndarray:
    """Return the 4 corners of the OBB projected onto the XY plane."""
    corners = obb_corners(box)               # (8,3)
    # Keep the bottom 4 (or just project all 8 and take unique hull)
    xy = corners[:, :2]
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(xy)
        return xy[hull.vertices]
    except Exception:
        return xy[:4]


# ══════════════════════════════════════════════
# 3.  ROOM GEOMETRY HELPERS
# ══════════════════════════════════════════════

def get_junction_coords(annotation: dict) -> dict:
    """id → np.array([x,y,z])"""
    return {j["ID"]: np.array(j["coordinate"]) for j in annotation["junctions"]}


def get_room_walls(annotation: dict, junction_coords: dict) -> list:
    """
    Returns a list of wall polygons as (N,3) arrays.
    A plane in Structured3D is defined by its normal + offset.
    We recover the polygon vertices from the planeLineMatrix and lineJunctionMatrix.
    """
    planes        = annotation["planes"]
    lines         = annotation["lines"]
    plane_line    = np.array(annotation["planeLineMatrix"])   # (P, L)
    line_junction = np.array(annotation["lineJunctionMatrix"])# (L, J)
    jc            = junction_coords

    wall_polys = []
    for plane in planes:
        pid  = plane["ID"]
        ptype = plane.get("type", "")
        # Collect line indices for this plane
        line_ids = np.where(plane_line[pid] > 0)[0].tolist()
        # Collect junction indices for those lines
        jun_set = set()
        for lid in line_ids:
            jids = np.where(line_junction[lid] > 0)[0]
            jun_set.update(jids.tolist())
        coords = np.array([jc[j] for j in sorted(jun_set)])
        if len(coords) >= 3:
            wall_polys.append({"type": ptype, "coords": coords, "plane_id": pid})
    return wall_polys


# ══════════════════════════════════════════════
# 4.  TOP-DOWN FLOOR-PLAN VISUALISATION
# ══════════════════════════════════════════════

PALETTE = {
    "floor":   "#e8e8e8",
    "ceiling": "#ccccff",
    "wall":    "#555555",
    "door":    "#ff9900",
    "window":  "#00ccff",
    "object":  "#ff4444",
}

def plot_floorplan(annotation: dict, bboxes: list, out_path: str):
    jc     = get_junction_coords(annotation)
    walls  = get_room_walls(annotation, jc)
    sem_map = {sid: s for s in annotation["semantics"] for sid in s.get("planeID", [])}

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")

    # ── Draw room planes (projected to XY) ──
    for wp in walls:
        xy   = wp["coords"][:, :2]
        ptype = wp["type"]
        pid   = wp["plane_id"]
        sem   = sem_map.get(pid, {}).get("type", ptype)

        color = PALETTE.get(sem, PALETTE.get(ptype, "#aaaaaa"))
        alpha = 0.15 if ptype in ("floor", "ceiling") else 0.4

        poly = plt.Polygon(xy, closed=True, facecolor=color,
                           edgecolor="#333333", linewidth=0.5, alpha=alpha)
        ax.add_patch(poly)

    # ── Draw object OBBs ──
    try:
        from scipy.spatial import ConvexHull
        have_scipy = True
    except ImportError:
        have_scipy = False

    for i, box in enumerate(bboxes):
        corners = obb_corners(box)
        xy = corners[:, :2]
        if have_scipy and len(xy) >= 3:
            try:
                hull = ConvexHull(xy)
                hull_xy = xy[hull.vertices]
            except Exception:
                hull_xy = xy[:4]
        else:
            hull_xy = xy[:4]

        poly = plt.Polygon(hull_xy, closed=True,
                           facecolor=PALETTE["object"], edgecolor="#880000",
                           linewidth=0.8, alpha=0.5)
        ax.add_patch(poly)

        # Label
        cx, cy = np.mean(hull_xy, axis=0)
        ax.text(cx, cy, str(box["ID"]), fontsize=5, ha="center", va="center", color="white")

    # ── Legend ──
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7)

    all_jx = [v[0] for v in jc.values()]
    all_jy = [v[1] for v in jc.values()]
    margin = 300
    ax.set_xlim(min(all_jx) - margin, max(all_jx) + margin)
    ax.set_ylim(min(all_jy) - margin, max(all_jy) + margin)
    ax.set_title(f"Top-Down Floor Plan — {SCENE_ID}", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[✓] Floor plan saved → {out_path}")


# ══════════════════════════════════════════════
# 5.  3-D MATPLOTLIB VISUALISATION (CPU, no GPU)
#     A lightweight alternative to Open3D
# ══════════════════════════════════════════════

def plot_3d_scene(annotation: dict, bboxes: list, out_path: str,
                  max_boxes: int = 50):
    jc    = get_junction_coords(annotation)
    walls = get_room_walls(annotation, jc)

    fig   = plt.figure(figsize=(16, 10))
    ax    = fig.add_subplot(111, projection="3d")

    # ── Room planes ──
    for wp in walls:
        coords = wp["coords"]
        ptype  = wp["type"]
        color  = {"floor": "lightgrey", "ceiling": "lightblue",
                   "wall": "wheat"}.get(ptype, "wheat")
        if len(coords) >= 3:
            verts = [coords.tolist()]
            poly  = Poly3DCollection(verts, alpha=0.25,
                                     facecolor=color, edgecolor="#888888",
                                     linewidth=0.3)
            ax.add_collection3d(poly)

    # ── Object OBBs ──
    EDGES = [
        (0,1),(1,2),(2,3),(3,0),   # bottom face
        (4,5),(5,6),(6,7),(7,4),   # top face
        (0,4),(1,5),(2,6),(3,7),   # verticals
    ]
    colors_cycle = plt.cm.tab20.colors
    for i, box in enumerate(bboxes[:max_boxes]):
        corners = obb_corners(box)            # (8,3)
        col     = colors_cycle[i % len(colors_cycle)]
        for e in EDGES:
            xs = [corners[e[0],0], corners[e[1],0]]
            ys = [corners[e[0],1], corners[e[1],1]]
            zs = [corners[e[0],2], corners[e[1],2]]
            ax.plot(xs, ys, zs, color=col, linewidth=0.8)

    ax.set_title(f"3D Scene — {SCENE_ID} (first {max_boxes} objects)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (height)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[✓] 3D view saved → {out_path}")


# ══════════════════════════════════════════════
# 6.  OPEN3D INTERACTIVE VIEW  (optional / Colab)
#     Requires: pip install open3d
#     On headless Colab use o3d.visualization.draw_plotly()
# ══════════════════════════════════════════════

def build_open3d_scene(annotation: dict, bboxes: list):
    try:
        import open3d as o3d
    except ImportError:
        print("[!] open3d not installed — skipping interactive view.")
        return None

    jc    = get_junction_coords(annotation)
    walls = get_room_walls(annotation, jc)
    geometries = []

    # ── Wall meshes via triangle fans ──
    for wp in walls:
        verts = wp["coords"] / 1000.0           # mm → m
        if len(verts) < 3:
            continue
        triangles = [[0, i, i+1] for i in range(1, len(verts)-1)]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        ptype = wp["type"]
        color = {"floor": [0.8,0.8,0.8], "ceiling": [0.7,0.7,1.0]}.get(ptype, [0.9,0.85,0.75])
        mesh.paint_uniform_color(color)
        geometries.append(mesh)

    # ── OBB geometries ──
    cmap = plt.cm.tab20.colors
    for i, box in enumerate(bboxes):
        c = np.array(box["centroid"]) / 1000.0
        h = np.array(box["coeffs"])   / 1000.0
        R = np.array(box["basis"])

        obb = o3d.geometry.OrientedBoundingBox()
        obb.center = c
        obb.extent = h * 2
        obb.R      = R.T                        # Open3D expects column-major
        obb.color  = cmap[i % len(cmap)][:3]
        geometries.append(obb)

    return geometries


# ══════════════════════════════════════════════
# 7.  OCCLUSION / VISIBILITY CHECK  (foundation)
#     Given:
#       - observer position (x,y,z)
#       - target position   (x,y,z)
#       - list of OBBs
#     Returns: (bool, list_of_occluding_box_ids)
# ══════════════════════════════════════════════

def ray_obb_intersect(ray_origin: np.ndarray, ray_dir: np.ndarray,
                      box: dict) -> bool:
    """
    Slab-method ray–OBB intersection test.
    ray_origin, ray_dir : (3,) arrays in scene units.
    box                 : one entry from bbox_3d.json.
    Returns True if ray hits the OBB.
    """
    c  = np.array(box["centroid"])
    h  = np.array(box["coeffs"])
    R  = np.array(box["basis"])    # rows = local axes

    # Transform ray to OBB local space
    d_local = R @ (ray_dir)
    o_local = R @ (ray_origin - c)

    t_min, t_max = -np.inf, np.inf
    for i in range(3):
        if abs(d_local[i]) < 1e-9:
            if abs(o_local[i]) > h[i]:
                return False
        else:
            t1 = (-h[i] - o_local[i]) / d_local[i]
            t2 = ( h[i] - o_local[i]) / d_local[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
    return t_max >= 0 and t_min <= t_max


def check_visibility(observer: list, target: list, bboxes: list,
                     observer_box_id: int = None, target_box_id: int = None
                     ) -> dict:
    """
    Check whether 'observer' has line-of-sight to 'target'.

    Parameters
    ----------
    observer        : [x, y, z]
    target          : [x, y, z]
    bboxes          : list of bbox_3d entries
    observer_box_id : OBB id that the observer is inside (skip self-intersection)
    target_box_id   : OBB id of the target object (skip target itself)

    Returns
    -------
    {
      "visible"       : bool,
      "occluders"     : [list of occluding box IDs],
      "ray_length_mm" : float,
    }
    """
    o   = np.array(observer, dtype=float)
    t   = np.array(target,   dtype=float)
    diff = t - o
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return {"visible": True, "occluders": [], "ray_length_mm": 0.0}

    ray_dir = diff / dist          # unit vector
    occluders = []

    for box in bboxes:
        bid = box["ID"]
        if bid == observer_box_id or bid == target_box_id:
            continue
        if ray_obb_intersect(o, ray_dir, box):
            # Confirm the intersection is BETWEEN observer and target
            # (not behind observer or past target)
            # A quick centroid-distance heuristic is enough for now
            c_box = np.array(box["centroid"])
            t_proj = np.dot(c_box - o, ray_dir)
            if 0 < t_proj < dist:
                occluders.append(bid)

    return {
        "visible":        len(occluders) == 0,
        "occluders":      occluders,
        "ray_length_mm":  dist,
    }


# ══════════════════════════════════════════════
# 8.  INSTANCE MASK ANALYSIS
#     Parse which pixel instance IDs appear &
#     link them back to bbox IDs if possible.
# ══════════════════════════════════════════════

def analyse_instance_mask(img_path: str) -> dict:
    img  = Image.open(img_path)
    arr  = np.array(img)           # (H, W) or (H, W, C)

    if arr.ndim == 3:
        # Instance ID is commonly encoded as R + G*256 (16-bit in 2 channels)
        instance_map = arr[:,:,0].astype(np.uint32) + arr[:,:,1].astype(np.uint32) * 256
    else:
        instance_map = arr.astype(np.uint32)

    unique_ids, counts = np.unique(instance_map, return_counts=True)
    result = {
        "unique_instance_ids": unique_ids.tolist(),
        "pixel_counts":        counts.tolist(),
        "image_shape":         arr.shape,
    }
    print(f"[✓] Instance mask: {arr.shape}, {len(unique_ids)} unique IDs found.")
    return result


# ══════════════════════════════════════════════
# 9.  SCENE GRAPH  (simple dict representation)
#     Nodes = objects; edges = spatial relations
# ══════════════════════════════════════════════

def build_scene_graph(bboxes: list, annotation: dict) -> dict:
    """
    Returns a lightweight scene graph:
      nodes: {box_id: {"centroid", "size_mm", "label"}}
      edges: {(id_a, id_b): {"relation": "occludes" | "near" | "above"}}
    """
    nodes = {}
    for box in bboxes:
        bid = box["ID"]
        nodes[bid] = {
            "centroid": box["centroid"],
            "size_mm":  [2*c for c in box["coeffs"]],   # full extents
            "label":    f"obj_{bid}",                    # replace with semantic class later
        }

    edges = {}
    ids   = [b["ID"] for b in bboxes]
    boxes = {b["ID"]: b for b in bboxes}

    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            ca = np.array(boxes[a]["centroid"])
            cb = np.array(boxes[b]["centroid"])
            dist = np.linalg.norm(ca[:2] - cb[:2])      # XY distance

            # "near" relation: within 1 m (1000 mm)
            if dist < 1000:
                edges[(a, b)] = {"relation": "near", "dist_mm": float(dist)}

            # "above" relation: same XY footprint, different Z
            if dist < 500 and abs(ca[2] - cb[2]) > 200:
                high, low = (a, b) if ca[2] > cb[2] else (b, a)
                edges[(high, low)] = {"relation": "above"}

    return {"nodes": nodes, "edges": edges}


# ══════════════════════════════════════════════
# 10.  STATEMENT VALIDATOR  (skeleton)
#      Called with a parsed statement like:
#        "person at [x,y,z] claims to see obj_id=42"
# ══════════════════════════════════════════════

def validate_statement(statement: dict, bboxes: list) -> dict:
    """
    statement = {
        "observer_pos": [x, y, z],   # mm
        "target_obj_id": int,
    }
    """
    obs_pos = statement["observer_pos"]
    tid     = statement["target_obj_id"]

    target_box = next((b for b in bboxes if b["ID"] == tid), None)
    if target_box is None:
        return {"valid": False, "reason": f"Object {tid} not found in scene."}

    target_pos = target_box["centroid"]
    result = check_visibility(obs_pos, target_pos, bboxes,
                              target_box_id=tid)

    if result["visible"]:
        verdict = "VALID — direct line of sight confirmed."
    else:
        verdict = (f"INVALID — line of sight blocked by objects: "
                   f"{result['occluders']}")

    return {
        "valid":     result["visible"],
        "verdict":   verdict,
        "occluders": result["occluders"],
        "distance_m": result["ray_length_mm"] / 1000.0,
    }


# ══════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  Object-Aware Scene Reconstruction  ")
    print(f"  Scene: {SCENE_ID}")
    print("=" * 55)

    # ── Load ──
    annotation = load_annotation(ANNOTATION_3D)
    bboxes     = load_bboxes(BBOX_3D)
    print(f"[✓] Loaded {len(bboxes)} OBBs, "
          f"{len(annotation['junctions'])} junctions, "
          f"{len(annotation['planes'])} planes.")

    # ── Floor plan ──
    plot_floorplan(annotation, bboxes,
                   out_path=os.path.join(OUTPUT_DIR, "floorplan.png"))

    # ── 3D matplotlib view ──
    plot_3d_scene(annotation, bboxes,
                  out_path=os.path.join(OUTPUT_DIR, "scene_3d.png"))

    # ── Scene graph ──
    sg = build_scene_graph(bboxes, annotation)
    print(f"[✓] Scene graph: {len(sg['nodes'])} nodes, {len(sg['edges'])} edges.")

    # ── Instance mask (panorama) ──
    if os.path.exists(INSTANCE_PANORAMA):
        mask_info = analyse_instance_mask(INSTANCE_PANORAMA)
    else:
        print(f"[!] Instance mask not found at {INSTANCE_PANORAMA}")

    # ── Example visibility check ──
    if len(bboxes) >= 2:
        obs_pos = bboxes[0]["centroid"]          # observer = centre of obj 0
        stmt = {
            "observer_pos": obs_pos,
            "target_obj_id": bboxes[-1]["ID"],   # claims to see last obj
        }
        verdict = validate_statement(stmt, bboxes)
        print("\n── Example Validation ──")
        print(f"  Observer position : {[round(v,1) for v in obs_pos]}")
        print(f"  Target object ID  : {stmt['target_obj_id']}")
        print(f"  Result            : {verdict['verdict']}")
        if verdict["occluders"]:
            print(f"  Occluding objects : {verdict['occluders']}")

    # ── Optional: Open3D interactive ──
    # Uncomment the following in a local environment with a display:
    # geoms = build_open3d_scene(annotation, bboxes)
    # if geoms:
    #     import open3d as o3d
    #     o3d.visualization.draw_geometries(geoms, window_name=SCENE_ID)
    # On headless Colab:
    # o3d.visualization.draw_plotly(geoms)   # renders in notebook cell

    print("\n[✓] All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()