import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle

# ==============================
# LOAD OBJECTS
# ==============================

with open("data/scene_00001/bbox_3d.json") as f:
    bbox_data = json.load(f)

objects = []

for obj in bbox_data:

    center = np.array(obj["centroid"]) / 1000
    size = np.array(obj["coeffs"]) * 2 / 1000

    objects.append({
        "center": center,
        "size": size
    })

print("Loaded objects:", len(objects))


# ==============================
# LOAD STRUCTURE LINES
# ==============================

layout_lines = []

try:

    with open("data/scene_00001/annotation_3d.json") as f:
        ann = json.load(f)

    for line in ann["lines"]:
        pt = np.array(line["point"]) / 1000
        layout_lines.append(pt)

    print("Loaded layout lines:", len(layout_lines))

except:
    print("No layout annotations")


# ==============================
# STATIC SCENE SETUP
# ==============================

fig, ax = plt.subplots(figsize=(8,8))

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

ax.set_title("Draw Suspect Path with Mouse")


# draw objects
rectangles = []

for obj in objects:

    cx, cy = obj["center"][:2]
    sx, sy = obj["size"][:2]

    rect = Rectangle(
        (cx - sx/2, cy - sy/2),
        sx,
        sy,
        edgecolor="blue",
        facecolor="none"
    )

    rectangles.append(rect)
    ax.add_patch(rect)


# draw layout
for pt in layout_lines:
    ax.plot(pt[0], pt[1], "k.", alpha=0.4)


# ==============================
# PATH RECORDING
# ==============================

path = []

def onclick(event):

    if event.xdata is None:
        return

    x = event.xdata
    y = event.ydata

    path.append([x,y])

    ax.plot(x,y,"ro",markersize=3)

    plt.draw()


cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()

path = np.array(path)

print("Recorded path points:", len(path))


# ==============================
# VISIBILITY CHECK
# ==============================

VIEW_RADIUS = 1.5

def visible_objects(actor_pos):

    visible = []

    for i,obj in enumerate(objects):

        cx, cy = obj["center"][:2]

        dist = np.linalg.norm(actor_pos - np.array([cx,cy]))

        if dist < VIEW_RADIUS:
            visible.append(i)

    return visible


# ==============================
# ANIMATION
# ==============================

fig2, ax2 = plt.subplots(figsize=(8,8))

ax2.set_xlim(-5,5)
ax2.set_ylim(-5,5)

ax2.set_title("Actor Movement + Visibility")


rects = []

for obj in objects:

    cx, cy = obj["center"][:2]
    sx, sy = obj["size"][:2]

    r = Rectangle(
        (cx - sx/2, cy - sy/2),
        sx,
        sy,
        edgecolor="blue",
        facecolor="none"
    )

    rects.append(r)
    ax2.add_patch(r)


actor = Circle((path[0][0],path[0][1]),0.15,color="red")
ax2.add_patch(actor)


def update(frame):

    pos = path[frame]

    actor.center = pos

    visible = visible_objects(pos)

    for i,r in enumerate(rects):

        if i in visible:
            r.set_edgecolor("red")
            r.set_linewidth(2)
        else:
            r.set_edgecolor("blue")
            r.set_linewidth(1)

    return rects + [actor]


ani = animation.FuncAnimation(
    fig2,
    update,
    frames=len(path),
    interval=100,
    blit=True
)

plt.show()