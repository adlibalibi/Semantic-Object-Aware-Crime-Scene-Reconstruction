# ============================================
# CRIME SCENE RECONSTRUCTION CORE MODULES
# ============================================

import json
import numpy as np
import cv2
import glob


# ============================================
# 1. LOAD SCENE GEOMETRY
# ============================================

def load_scene(scene_path):

    with open(scene_path + "/bbox_3d.json") as f:
        bbox_data = json.load(f)

    objects = []

    for obj in bbox_data:

        center = np.array(obj["centroid"]) / 1000
        size = np.array(obj["coeffs"]) * 2 / 1000

        if size[0] * size[1] < 0.05:
            continue

        objects.append({
            "center": center,
            "size": size
        })

    return objects


# ============================================
# 2. LOAD PANORAMA
# ============================================

def load_panorama(scene_path):

    pan_files = glob.glob(scene_path + "/**/panorama.png", recursive=True)

    pano = cv2.imread(pan_files[0])
    pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)

    H, W, _ = pano.shape

    return pano, W, H


# ============================================
# 3. COMPUTE MOVEMENT DIRECTION
# ============================================

def compute_angle(p1, p2):

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    ang = np.degrees(np.arctan2(dy, dx))

    if ang < 0:
        ang += 360

    return ang


# ============================================
# 4. EXTRACT POV FROM PANORAMA
# ============================================

def get_pov(panorama, angle, W, FOV=90):

    center = int((angle / 360) * W)
    width = int((FOV / 360) * W)

    left = center - width // 2
    right = center + width // 2

    if left < 0:
        view = np.hstack((panorama[:, left:], panorama[:, :right]))

    elif right > W:
        view = np.hstack((panorama[:, left:], panorama[:, :right - W]))

    else:
        view = panorama[:, left:right]

    return view


# ============================================
# 5. LINE OF SIGHT ENGINE
# ============================================

def line_intersects_rect(p1, p2, rect_center, rect_size):

    cx, cy = rect_center
    sx, sy = rect_size

    left = cx - sx / 2
    right = cx + sx / 2
    bottom = cy - sy / 2
    top = cy + sy / 2

    for t in np.linspace(0, 1, 20):

        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])

        if left < x < right and bottom < y < top:
            return True

    return False


# ============================================
# 6. OBJECT VISIBILITY CHECK
# ============================================

def is_visible(actor_pos, obj_index, objects):

    obj = objects[obj_index]
    obj_center = obj["center"][:2]

    for i, other in enumerate(objects):

        if i == obj_index:
            continue

        if line_intersects_rect(
            actor_pos,
            obj_center,
            other["center"][:2],
            other["size"][:2]
        ):
            return False

    return True


# ============================================
# 7. OBJECT INTERACTION DETECTION
# ============================================

def detect_object_interactions(frame, actor_pos, objects, event_buffer):

    for i, obj in enumerate(objects):

        center = obj["center"][:2]
        dist = np.linalg.norm(actor_pos - center)

        if not is_visible(actor_pos, i, objects):
            continue

        if dist < 0.5:
            event_buffer.append((frame, "touch", i))

        elif dist < 1.5:
            event_buffer.append((frame, "approach", i))

        elif dist < 3:
            event_buffer.append((frame, "observe", i))


# ============================================
# 8. TIMELINE BUILDER
# ============================================

def build_timeline(events):

    events = sorted(events, key=lambda x: x[0])

    timeline = []

    for e in events:
        if e not in timeline:
            timeline.append(e)

    return timeline


# ============================================
# 9. EXPLANATION ENGINE
# ============================================

def explain_crime(timeline):

    explanations = []

    for frame, etype, obj in timeline:

        if etype == "observe":
            explanations.append(
                f"Frame {frame}: Actor observed object {obj}"
            )

        if etype == "approach":
            explanations.append(
                f"Frame {frame}: Actor approached object {obj}"
            )

        if etype == "touch":
            explanations.append(
                f"Frame {frame}: Actor interacted with object {obj}"
            )

    return explanations


# ============================================
# 10. MAIN PIPELINE
# ============================================

def run_simulation(scene_path, path):

    objects = load_scene(scene_path)

    panorama, W, H = load_panorama(scene_path)

    event_buffer = []

    for frame in range(len(path) - 1):

        p1 = np.array(path[frame])
        p2 = np.array(path[frame + 1])

        angle = compute_angle(p1, p2)

        pov = get_pov(panorama, angle, W)

        detect_object_interactions(
            frame,
            p1,
            objects,
            event_buffer
        )

    timeline = build_timeline(event_buffer)

    explanation = explain_crime(timeline)

    return timeline, explanation