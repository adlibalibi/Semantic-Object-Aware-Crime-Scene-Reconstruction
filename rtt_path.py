import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ==============================
# 1. LOAD DATA
# ==============================

with open("data/scene_00001/bbox_3d.json") as f:
    bboxes = json.load(f)

objects = []

for obj in bboxes:
    centroid = np.array(obj["centroid"]) / 1000.0
    coeffs = np.array(obj["coeffs"]) / 1000.0

    objects.append({
        "id": obj.get("id", "obj"),
        "center": centroid,
        "width": coeffs[0]*2,
        "depth": coeffs[1]*2,
        "height": coeffs[2]*2
    })

# ==============================
# 2. FILTER
# ==============================

filtered_objects = []

for obj in objects:
    if obj["center"][2] > 2.5:
        continue
    if obj["width"] < 0.2 or obj["depth"] < 0.2:
        continue
    filtered_objects.append(obj)

print("Filtered objects:", len(filtered_objects))

# ==============================
# 3. NORMALIZE
# ==============================

xs = [o["center"][0] for o in filtered_objects]
ys = [o["center"][1] for o in filtered_objects]

min_x = min(xs)
min_y = min(ys)

for o in filtered_objects:
    o["center"][0] -= min_x
    o["center"][1] -= min_y

scene_width = max(xs) - min_x
scene_height = max(ys) - min_y

# ==============================
# 4. ACTOR
# ==============================

class Actor:
    def __init__(self, x, y, orientation=0, fov=60):
        self.position = np.array([x, y])
        self.orientation = orientation
        self.fov = fov

actor = Actor(1.0, 1.0, orientation=0)

# ==============================
# 5. SAFE WALKING PATH
# ==============================

# perimeter-style exploration path (collision free)

path = []

margin = 0.7

points = [
    (margin, margin),
    (scene_width-margin, margin),
    (scene_width-margin, scene_height-margin),
    (margin, scene_height-margin),
    (margin, margin)
]

for i in range(len(points)-1):

    p1 = np.array(points[i])
    p2 = np.array(points[i+1])

    steps = 80

    for t in np.linspace(0,1,steps):
        pos = p1*(1-t) + p2*t
        path.append(pos)

path = np.array(path)

# ==============================
# 6. VISIBILITY COUNTER
# ==============================

visibility_counter = {i:0 for i in range(len(filtered_objects))}

# ==============================
# 7. GEOMETRY UTILITIES
# ==============================

def lines_intersect(p1, p2, q1, q2):

    def ccw(a,b,c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

    return (ccw(p1,q1,q2) != ccw(p2,q1,q2)) and \
           (ccw(p1,p2,q1) != ccw(p1,p2,q2))

def line_intersects_rect(p1, p2, rect_center, width, depth):

    rx, ry = rect_center[:2]

    left = rx-width/2
    right = rx+width/2
    bottom = ry-depth/2
    top = ry+depth/2

    edges = [
        ((left,bottom),(right,bottom)),
        ((right,bottom),(right,top)),
        ((right,top),(left,top)),
        ((left,top),(left,bottom))
    ]

    for edge in edges:
        if lines_intersect(p1,p2,edge[0],edge[1]):
            return True

    return False

# ==============================
# 8. VISIBILITY FUNCTION
# ==============================

def is_visible(actor, obj, objects):

    obj_pos = obj["center"][:2]

    direction = np.array([
        np.cos(np.radians(actor.orientation)),
        np.sin(np.radians(actor.orientation))
    ])

    to_obj = obj_pos - actor.position
    dist = np.linalg.norm(to_obj)

    if dist == 0:
        return True

    unit = to_obj / dist

    angle = np.degrees(np.arccos(
        np.clip(np.dot(direction, unit), -1, 1)
    ))

    if angle > actor.fov/2:
        return False

    for other in objects:

        if other is obj:
            continue

        if line_intersects_rect(actor.position,
                                obj_pos,
                                other["center"],
                                other["width"],
                                other["depth"]):

            other_dist = np.linalg.norm(
                other["center"][:2] - actor.position
            )

            if other_dist < dist:
                return False

    return True

# ==============================
# 9. FIGURE 1 – STATIC SCENE
# ==============================

fig1, ax1 = plt.subplots(figsize=(8,8))

for idx,obj in enumerate(filtered_objects):

    x,y = obj["center"][:2]
    w = obj["width"]
    d = obj["depth"]

    rect = Rectangle(
        (x-w/2,y-d/2),
        w,d,
        fill=False,
        edgecolor="black"
    )

    ax1.add_patch(rect)
    ax1.text(x,y,str(idx),fontsize=6)

ax1.plot(actor.position[0],actor.position[1],"bo")

ax1.set_xlim(0,scene_width)
ax1.set_ylim(0,scene_height)
ax1.set_aspect("equal")
ax1.set_title("Figure 1: Static Scene Layout")

legend = [
    Line2D([0],[0],color="black",label="Object"),
    Line2D([0],[0],marker="o",color="w",
           markerfacecolor="blue",label="Actor")
]

ax1.legend(handles=legend)

plt.show()

# ==============================
# 10. FIGURE 2 – ANIMATION
# ==============================

fig2,ax2 = plt.subplots(figsize=(8,8))

def update(frame):

    ax2.clear()

    actor.position = path[frame]
    actor.orientation = frame*4

    for idx,obj in enumerate(filtered_objects):

        visible = is_visible(actor,obj,filtered_objects)

        if visible:
            visibility_counter[idx]+=1

        x,y = obj["center"][:2]
        w = obj["width"]
        d = obj["depth"]

        rect = Rectangle(
            (x-w/2,y-d/2),
            w,d,
            fill=False,
            edgecolor="red" if visible else "gray"
        )

        ax2.add_patch(rect)
        ax2.text(x,y,str(idx),fontsize=6)

    ax2.plot(path[:,0],path[:,1],"g--",alpha=0.3)

    ax2.plot(actor.position[0],actor.position[1],"bo")

    # FOV

    L = 2

    left = np.radians(actor.orientation-actor.fov/2)
    right = np.radians(actor.orientation+actor.fov/2)

    ax2.plot(
        [actor.position[0],actor.position[0]+L*np.cos(left)],
        [actor.position[1],actor.position[1]+L*np.sin(left)],
        "b--"
    )

    ax2.plot(
        [actor.position[0],actor.position[0]+L*np.cos(right)],
        [actor.position[1],actor.position[1]+L*np.sin(right)],
        "b--"
    )

    ax2.set_xlim(0,scene_width)
    ax2.set_ylim(0,scene_height)
    ax2.set_aspect("equal")
    ax2.set_title("Figure 2: Object Visibility Simulation")

ani = FuncAnimation(
    fig2,
    update,
    frames=len(path),
    interval=80
)

plt.show()

# ==============================
# 11. VISIBILITY HEATMAP
# ==============================

fig3,ax3 = plt.subplots(figsize=(8,8))

max_vis = max(visibility_counter.values())

for idx,obj in enumerate(filtered_objects):

    x,y = obj["center"][:2]
    w = obj["width"]
    d = obj["depth"]

    intensity = visibility_counter[idx]/max_vis

    rect = Rectangle(
        (x-w/2,y-d/2),
        w,d,
        color=(1,0,0,intensity)
    )

    ax3.add_patch(rect)

ax3.set_xlim(0,scene_width)
ax3.set_ylim(0,scene_height)
ax3.set_aspect("equal")
ax3.set_title("Figure 3: Visibility Heatmap")

plt.show()