import matplotlib
matplotlib.use("TkAgg")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge
import cv2
import glob

from ultralytics import YOLO


# ==============================
# LOAD YOLO MODEL
# ==============================

yolo = YOLO("yolov8n.pt")


# ==============================
# LOAD 3D OBJECTS
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
# DRAW PATH
# ==============================

fig, ax = plt.subplots(figsize=(8,8))

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

ax.set_title("Draw Actor Path")

for obj in objects:

    cx, cy = obj["center"][:2]
    sx, sy = obj["size"][:2]

    r = Rectangle((cx-sx/2,cy-sy/2),sx,sy,
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
VIEW_DIST = 5


def compute_angle(p1,p2):

    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]

    ang = np.degrees(np.arctan2(dy,dx))

    if ang < 0:
        ang += 360

    return ang


# ==============================
# PANORAMA POV
# ==============================

def get_pov(angle):

    center = int((angle/360)*W)

    width = int((FOV/360)*W)

    left = center - width//2
    right = center + width//2

    if left < 0:

        view = np.hstack((panorama[:,left:],panorama[:,:right]))

    elif right > W:

        view = np.hstack((panorama[:,left:],panorama[:,:right-W]))

    else:

        view = panorama[:,left:right]

    return view


# ==============================
# YOLO DETECTION
# ==============================

def detect_objects(frame):

    results = yolo(frame)

    detections = []

    for r in results:

        boxes = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()

        for box,c,p in zip(boxes,conf,cls):

            if c < 0.4:
                continue

            x1,y1,x2,y2 = map(int,box)

            label = yolo.names[int(p)]

            detections.append({
                "label":label,
                "box":(x1,y1,x2,y2),
                "conf":float(c)
            })

    return detections


# ==============================
# EVENT MEMORY
# ==============================

event_buffer = []


def record_events(frame,detections):

    for d in detections:

        label = d["label"]

        if label in ["knife","scissors","bottle"]:
            event_buffer.append((frame,"weapon_visible"))

        if label == "person":
            event_buffer.append((frame,"person_visible"))

        if label in ["chair","table"]:
            event_buffer.append((frame,"furniture_nearby"))


# ==============================
# TIMELINE ENGINE
# ==============================

def build_timeline(events):

    timeline = []

    last_event = None

    for f,e in events:

        if last_event == e:
            continue

        timeline.append((f,e))

        last_event = e

    return timeline


# ==============================
# EXPLANATION ENGINE
# ==============================

def explain_crime(timeline):

    weapon = None
    victim = None

    for f,e in timeline:

        if e == "weapon_visible" and weapon is None:
            weapon = f

        if e == "person_visible":
            victim = f

    print("\n--- Crime Scene Interpretation ---\n")

    if weapon:
        print("Possible weapon observed at frame",weapon)

    if victim:
        print("Another person observed at frame",victim)

    if weapon and victim and victim > weapon:
        print("Potential confrontation after weapon acquisition")

    if weapon and not victim:
        print("Actor carried weapon but no victim detected")

    print("\nTimeline:\n")

    for t in timeline:
        print(t)


# ==============================
# ANIMATION
# ==============================

fig2 = plt.figure(figsize=(12,6))

ax_scene = fig2.add_subplot(121)
ax_pov = fig2.add_subplot(122)

ax_scene.set_xlim(-5,5)
ax_scene.set_ylim(-5,5)

for obj in objects:

    cx,cy = obj["center"][:2]
    sx,sy = obj["size"][:2]

    r = Rectangle((cx-sx/2,cy-sy/2),sx,sy,
                  edgecolor="blue",
                  facecolor="none")

    ax_scene.add_patch(r)

actor = Circle(path[0],0.15,color="red")

ax_scene.add_patch(actor)

cone = Wedge(path[0],3,0,FOV,color="yellow",alpha=0.3)

ax_scene.add_patch(cone)

pov_img = ax_pov.imshow(np.zeros((100,100,3),dtype=np.uint8))


def update(frame):

    if frame >= len(path)-1:
        return

    p1 = path[frame]
    p2 = path[frame+1]

    actor.center = p1

    angle = compute_angle(p1,p2)

    cone.set_center(p1)
    cone.theta1 = angle-FOV/2
    cone.theta2 = angle+FOV/2

    pov = get_pov(angle)

    detections = detect_objects(pov)

    record_events(frame,detections)

    for d in detections:

        x1,y1,x2,y2 = d["box"]

        cv2.rectangle(pov,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            pov,
            d["label"],
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

    pov_img.set_data(pov)

    return actor,pov_img


ani = animation.FuncAnimation(
    fig2,
    update,
    frames=len(path)-1,
    interval=200
)

plt.show()


# ==============================
# BUILD TIMELINE
# ==============================

timeline = build_timeline(event_buffer)

explain_crime(timeline)