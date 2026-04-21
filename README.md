# 🔍 ForenSim — Forensic Crime Scene Path Validator

> A computer vision project for reconstructing and validating witness narratives against 3D scene geometry, built on the [Structured3D](https://structured3d-dataset.org/) dataset.

---

## Overview

ForenSim is an interactive forensic tool that reconstructs indoor crime scenes from 3D annotations and validates whether a described movement path is **physically plausible** given the actual layout of a room. It answers the core forensic question:

> *"Could the person have actually moved the way the witness described?"*

If a witness says a suspect walked from the kitchen counter past the sofa to the north window — ForenSim checks every step of that path against the room's real walls, furniture, and obstacles, and tells you whether the statement holds up spatially.

---

## Demo

![ForenSim Validator Interface](forensim_validator.html)

The app renders a top-down 2D floorplan of the scene. You click to lay down waypoints tracing the actor's path, hit **Validate**, and get an instant spatial plausibility report.

---

## Features

- **Top-down 2D Floorplan** rendered from real 3D bounding box annotations
- **Waypoint path drawing** — click to trace any actor path directly on the scene map
- **Height-aware collision detection** — objects are only considered blocking if they fall within the actor's body height range (passable floor mats ≠ impassable walls)
- **Per-segment pass/fail analysis** with collision coordinates
- **Plausibility Score (0–100)** summarising how feasible the path is
- **Object interaction log** — proximity and contact events along the path
- **Visual collision markers** with pulsing animation on blocked segments
- **Preloaded example scenarios** (valid paths, suspicious wall-bypass attempt, kitchen crossing, etc.)
- **Pan + zoom** on the scene canvas
- **Scale bar** and live coordinate display

---

## Project Structure

```
forensim/
│
├── forensim_validator.html        # ← Main interactive app (self-contained, open in browser)
│
├── wetryagain.py                  # Scene reconstruction pipeline (Python backend)
│   ├── load_annotation()          # Parses annotation_3d.json (junctions, planes, lines)
│   ├── load_bboxes()              # Loads bbox_3d.json OBBs
│   ├── obb_corners()              # Computes 8 corners of oriented bounding boxes
│   ├── plot_floorplan()           # Top-down 2D floor plan with objects (matplotlib)
│   ├── plot_3d_scene()            # 3D matplotlib view of room + objects
│   ├── build_open3d_scene()       # Optional Open3D interactive 3D render
│   ├── ray_obb_intersect()        # Ray vs OBB intersection (slab method)
│   ├── check_visibility()         # Line-of-sight check between two 3D points
│   ├── analyse_instance_mask()    # Parses instance segmentation masks
│   ├── build_scene_graph()        # Scene graph with spatial relations (near, above)
│   └── validate_statement()       # Validates observer → target visibility claim
│
├── panoroma_with_timeline.py      # Animated actor path + panoramic POV viewer
│   ├── Path drawing UI            # Click-to-draw actor trajectory (matplotlib)
│   ├── get_pov()                  # Crops panorama to simulated FOV at each waypoint
│   ├── is_visible()               # Occlusion check per object along path
│   ├── detect_object_interactions() # Classifies touch / approach / observe events
│   ├── build_timeline()           # Ordered event log from path traversal
│   ├── detect_suspicious_patterns() # Flags observe-then-touch sequences
│   └── explain_crime()            # Prints plain-language narrative from timeline
│
├── semantic.py                    # Multi-view semantic labelling pipeline
│   ├── run_yolo_multiview()       # YOLOv8 inference on 5 perspective renders
│   ├── run_yolo_panorama()        # YOLOv8 inference on panoramic render
│   ├── clean_detections()         # Filters invalid/irrelevant classes
│   ├── merge_detections()         # IoU-based NMS across views
│   ├── assign_labels()            # Maps YOLO detections → 3D bounding boxes
│   ├── heuristic_label()          # Fallback labelling by object dimensions
│   └── draw_floorplan()           # Saves semantically labelled floor plan
│
├── data/scene_00001/
│   ├── annotation_3d.json         # Room geometry (junctions, planes, lines, semantics)
│   ├── bbox_3d.json               # 3D oriented bounding boxes for all scene objects
│   ├── panorama.png               # Equirectangular instance segmentation panorama
│   ├── perspective_1.png          # Perspective view 0 (instance mask)
│   ├── perspective_2.png          # Perspective view 1
│   ├── perspective_3.png          # Perspective view 2
│   ├── perspective_4.png          # Perspective view 3
│   └── perspective_5.png          # Perspective view 4
│
└── outputs/
    ├── floorplan.png              # Generated top-down floor plan
    └── scene_3d.png               # Generated 3D matplotlib render
```

---

## Dataset

This project uses the **[Structured3D](https://structured3d-dataset.org/)** dataset — a large-scale photo-realistic dataset with detailed structured annotations for 3D scene understanding.

Specifically used from `scene_00001`:

| File | Contents |
|---|---|
| `annotation_3d.json` | Room geometry: junctions, planes, lines, semantic labels |
| `bbox_3d.json` | 260 oriented bounding boxes (OBBs) with centroid, half-extents, rotation basis — all in millimetres |
| `panorama/full/instance.png` | Full equirectangular instance segmentation render |
| `perspective/full/{0–4}/instance.png` | 5 perspective instance renders from different viewpoints |

### Coordinate System

Structured3D uses **millimetres** for all coordinates. The validator converts to **metres** (`value / 1000`) for display. The scene spans approximately:
- X: −3.9 m to +3.5 m
- Y: −6.3 m to +5.5 m
- Z: 0 m to ~4.6 m (floor to ceiling)

---

## How It Works

### 1. Scene Parsing (`wetryagain.py`)

The `annotation_3d.json` provides room geometry as a topological graph of **junctions** (vertices), **lines** (edges), and **planes** (faces). The script:
1. Reconstructs wall polygons by tracing the `planeLineMatrix` → `lineJunctionMatrix` connectivity
2. Loads all 260 OBBs from `bbox_3d.json`
3. Projects everything onto the XY plane for the 2D floor plan
4. Optionally builds a full Open3D interactive 3D scene

### 2. Collision Detection (Validator App)

The HTML validator uses a **parametric ray-vs-AABB slab test** in 2D (XY plane):

```
For each path segment (p1 → p2):
  For each scene object:
    1. Check if object is blocking at actor's height (Z overlap test)
    2. Expand object footprint by actor radius (0.18m)
    3. Compute tMin/tMax intersection with expanded AABB
    4. If tMin ≤ tMax → COLLISION
```

Objects with a Z extent that doesn't overlap the actor's body (0.1m – actor_height) are treated as passable (e.g. a floor rug, a ceiling fixture).

### 3. Semantic Labelling (`semantic.py`)

YOLOv8m runs inference on all 5 perspective renders + the panorama. Detections are:
1. Filtered (remove vehicles, people, outdoor objects)
2. Merged across views using IoU-based NMS
3. Matched to 3D bounding boxes via a 2D projection + IoU scoring
4. Any unmatched box falls back to **heuristic labelling** based on physical dimensions (e.g. objects with height > 1.5m → `wall/column`, large flat objects → `table`)

### 4. Timeline + POV (`panoroma_with_timeline.py`)

An animated dual-panel view shows:
- **Left panel**: top-down scene with actor position and FOV cone
- **Right panel**: cropped panoramic POV at each waypoint

The event engine classifies each object encounter along the path as `observe` (>1.5m away), `approach` (<1.5m), or `touch` (<0.5m), building a forensic timeline. It flags suspicious sequences like `observe → touch` (pre-planned interaction).

---

## Installation

### Python Scripts

```bash
pip install numpy matplotlib pillow open3d scipy ultralytics
```

For Colab:
```bash
!pip install open3d-python ultralytics
```

### HTML Validator

No installation needed. Just open `forensim_validator.html` in any modern browser. It is fully self-contained — all scene data is embedded as JSON inside the file.

---

## Usage

### Running the Python Pipeline

```python
# Edit the BASE path in wetryagain.py to point to your data folder
BASE = r"path/to/your/Structured3D/data"
SCENE_ID = "scene_00001"

python wetryagain.py
```

Outputs saved to `outputs/`:
- `floorplan.png` — top-down 2D floor plan with all objects labelled
- `scene_3d.png` — 3D matplotlib render

### Running the Semantic Labeller

```python
# Edit BASE path in semantic.py
python semantic.py
# Output: outputs/floorplan.png with semantic labels
```

### Using the Interactive Validator

1. Open `forensim_validator.html` in a browser
2. Read or type the witness statement in the left panel
3. Click on the scene map to place waypoints tracing the actor's path
   - Left-click: add waypoint
   - Right-click: remove last waypoint
   - Scroll: zoom in/out
   - Middle-mouse drag: pan
4. Optionally adjust the actor's height (affects what counts as blocking)
5. Click **VALIDATE PATH**
6. Read the Plausibility Score and per-step breakdown in the right panel

Or load one of the **4 preset scenarios** to see examples immediately.

---

## Example Output

### Valid Path
```
Score: 95 / 100
✓ STATEMENT PLAUSIBLE
All 4 movement segments clear of obstacles.
Step 1: Clear path (1.20m)
Step 2: Clear path (0.85m)
Step 3: Clear path (1.40m)
Step 4: Clear path (0.65m)
```

### Invalid Path (wall bypass attempt)
```
Score: 20 / 100
✗ STATEMENT IMPLAUSIBLE
Path contains 3 collision(s) with room objects.

⚠ WALL PANEL (obj #167): Path intersects at approx. (-2.29, -0.12).
⚠ WALL PANEL (obj #184): Path intersects at approx. (-1.80, -0.50).
⚠ CABINET/WARDROBE (obj #107): Path intersects at approx. (-0.72, -1.37).
```

---

## Key Modules at a Glance

| Module | Purpose |
|---|---|
| `obb_corners()` | Converts OBB (centroid + half-extents + rotation) to 8 world-space corners |
| `ray_obb_intersect()` | Slab-method ray-OBB intersection for line-of-sight queries |
| `check_visibility()` | Full 3D LOS check: observer → target, returns occluding object IDs |
| `validate_statement()` | High-level API: takes `{observer_pos, target_obj_id}` → returns verdict |
| `build_scene_graph()` | Builds spatial relationship graph (near / above) over all objects |
| `detect_suspicious_patterns()` | Detects forensically suspicious sequences in event timeline |

---

## Limitations & Future Work

- **OBB rotation**: The validator currently uses AABB (axis-aligned) approximations for collision. Full OBB rotation support is implemented in `wetryagain.py` (slab method) but not yet ported to the HTML validator.
- **Multi-floor scenes**: The Z-based height filter handles single-floor rooms; staircase logic is not implemented.
- **Semantic labels**: The heuristic labeller is approximate. Full YOLO-based semantic labelling requires the Python pipeline (`semantic.py`) to be run separately.
- **NLP statement parsing**: Currently the witness statement text is manual/decorative. A future version could parse natural language to auto-generate waypoints.
- **Multiple actors**: Currently supports one actor path at a time.

---

## Acknowledgements

- [Structured3D Dataset](https://structured3d-dataset.org/) — Zheng et al., ECCV 2020
- [YOLOv8 / Ultralytics](https://github.com/ultralytics/ultralytics) — used in `semantic.py`
- [Open3D](http://www.open3d.org/) — 3D visualisation backend

---
