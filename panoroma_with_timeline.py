import matplotlib
matplotlib.use("TkAgg")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge
import cv2
import glob


# ==============================
# LOAD OBJECT GEOMETRY
# ==============================

with open("data/scene_00001/bbox_3d.json") as f:
    bbox_data = json.load(f)

objects = []

for obj in bbox_data:

    center = np.array(obj["centroid"]) / 1000
    size = np.array(obj["coeffs"]) * 2 / 1000

    # ignore tiny objects
    if size[0] * size[1] < 0.05:
        continue

    objects.append({
        "center": center,
        "size": size
    })

print("Objects loaded:", len(objects))


# ==============================
# LOAD PANORAMA
# ==============================

pan_files = glob.glob("data/scene_00001/**/panorama.png", recursive=True)

panorama = cv2.imread(pan_files[0])
panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

H, W, _ = panorama.shape

print("Panorama size:", W, H)


# ==============================
# DRAW ACTOR PATH
# ==============================

fig, ax = plt.subplots(figsize=(8,8))

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

ax.set_title("Click to draw actor path")

for obj in objects:

    cx, cy = obj["center"][:2]
    sx, sy = obj["size"][:2]

    r = Rectangle((cx-sx/2, cy-sy/2),
                  sx, sy,
                  edgecolor="blue",
                  facecolor="none")

    ax.add_patch(r)

path = []

def onclick(event):

    if event.xdata is None:
        return

    path.append([event.xdata,event.ydata])

    ax.plot(event.xdata,event.ydata,"ro")

    plt.draw()

fig.canvas.mpl_connect("button_press_event",onclick)

plt.show(block=True)

path = np.array(path)


# ==============================
# CAMERA SETTINGS
# ==============================

FOV = 90


def compute_angle(p1,p2):

    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]

    ang = np.degrees(np.arctan2(dy,dx))

    if ang < 0:
        ang += 360

    return ang


# ==============================
# POV FROM PANORAMA
# ==============================

def get_pov(angle):

    center = int((angle/360)*W)

    width = int((FOV/360)*W)

    left = center - width//2
    right = center + width//2

    if left < 0:
        view = np.hstack((panorama[:,left:], panorama[:,:right]))

    elif right > W:
        view = np.hstack((panorama[:,left:], panorama[:,:right-W]))

    else:
        view = panorama[:,left:right]

    return view


# ==============================
# LINE OF SIGHT ENGINE
# ==============================

def line_intersects_rect(p1,p2,rect_center,rect_size):

    cx,cy = rect_center
    sx,sy = rect_size

    left = cx - sx/2
    right = cx + sx/2
    bottom = cy - sy/2
    top = cy + sy/2

    for t in np.linspace(0,1,20):

        x = p1[0] + t*(p2[0]-p1[0])
        y = p1[1] + t*(p2[1]-p1[1])

        if left < x < right and bottom < y < top:
            return True

    return False


def is_visible(actor_pos,obj_index):

    obj = objects[obj_index]

    obj_center = obj["center"][:2]

    for i,other in enumerate(objects):

        if i == obj_index:
            continue

        if line_intersects_rect(actor_pos,
                                obj_center,
                                other["center"][:2],
                                other["size"][:2]):

            return False

    return True


# ==============================
# EVENT MEMORY
# ==============================

event_buffer = []


def detect_object_interactions(frame,actor_pos):

    for i,obj in enumerate(objects):

        center = obj["center"][:2]

        dist = np.linalg.norm(actor_pos-center)

        if not is_visible(actor_pos,i):
            continue

        if dist < 0.5:

            event_buffer.append((frame,"touch",i))

        elif dist < 1.5:

            event_buffer.append((frame,"approach",i))

        elif dist < 3:

            event_buffer.append((frame,"observe",i))


# ==============================
# TIMELINE BUILDER
# ==============================

def build_timeline(events):

    events = sorted(events,key=lambda x:x[0])

    timeline = []

    for e in events:
        if e not in timeline:
            timeline.append(e)

    return timeline


# ==============================
# SUSPICIOUS PATTERN DETECTION
# ==============================

def detect_suspicious_patterns(timeline):

    suspicious = []

    for i in range(len(timeline)-1):

        f1,e1,obj1 = timeline[i]
        f2,e2,obj2 = timeline[i+1]

        if e1=="observe" and e2=="touch":

            suspicious.append(
                f"Actor inspected object {obj1} before touching it"
            )

    return suspicious


# ==============================
# EXPLANATION ENGINE
# ==============================

def explain_crime(timeline):

    print("\n----- Crime Scene Explanation -----\n")

    for frame,etype,obj in timeline:

        if etype=="observe":
            print(f"Frame {frame}: Actor observed object {obj}")

        if etype=="approach":
            print(f"Frame {frame}: Actor approached object {obj}")

        if etype=="touch":
            print(f"Frame {frame}: Actor interacted with object {obj}")

    suspicious = detect_suspicious_patterns(timeline)

    if suspicious:

        print("\n--- Suspicious Behaviour ---")

        for s in suspicious:
            print(s)


# ==============================
# ANIMATION SETUP
# ==============================

fig2 = plt.figure(figsize=(14,6))

ax_scene = fig2.add_subplot(121)
ax_pov = fig2.add_subplot(122)

ax_scene.set_xlim(-5,5)
ax_scene.set_ylim(-5,5)

for obj in objects:

    cx,cy = obj["center"][:2]
    sx,sy = obj["size"][:2]

    r = Rectangle((cx-sx/2,cy-sy/2),
                  sx,sy,
                  edgecolor="blue",
                  facecolor="none")

    ax_scene.add_patch(r)

actor = Circle(path[0],0.15,color="red")
ax_scene.add_patch(actor)

cone = Wedge(path[0],3,0,FOV,color="yellow",alpha=0.3)
ax_scene.add_patch(cone)

pov_img = ax_pov.imshow(np.zeros((100,100,3),dtype=np.uint8))


# ==============================
# ANIMATION LOOP
# ==============================

def update(frame):

    if frame >= len(path)-1:
        return

    p1 = path[frame]
    p2 = path[frame+1]

    actor.center = p1

    angle = compute_angle(p1,p2)

    cone.set_center(p1)
    cone.theta1 = angle - FOV/2
    cone.theta2 = angle + FOV/2

    pov = get_pov(angle)

    detect_object_interactions(frame,np.array(p1))

    pov_img.set_data(pov)

    return actor,pov_img


ani = animation.FuncAnimation(
    fig2,
    update,
    frames=len(path)-1,
    interval=300
)

plt.show()


# ==============================
# BUILD TIMELINE
# ==============================

timeline = build_timeline(event_buffer)

explain_crime(timeline)