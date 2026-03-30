import matplotlib
matplotlib.use("TkAgg")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge
import cv2
import glob
import torch
import warnings

# Suppress deprecation warnings from timm
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")


# ==============================
# LOAD MiDaS DEPTH MODEL
# ==============================

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform


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
# DEPTH ESTIMATION
# ==============================

def estimate_depth(image):

    img = transform(image)

    with torch.no_grad():
        depth = midas(img)

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()

    depth = depth.cpu().numpy()

    depth = (depth - depth.min())/(depth.max()-depth.min())

    return depth


depth_map = estimate_depth(panorama)


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
# IMPROVED SILHOUETTE DETECTION
# ==============================

def detect_silhouettes(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # improve contrast
    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    _, th = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)

    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    contours,_ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for c in contours:

        area = cv2.contourArea(c)

        if area < 200:
            continue

        x,y,w,h = cv2.boundingRect(c)

        aspect = w/h if h>0 else 0

        if aspect < 0.1 or aspect > 10:
            continue

        detections.append((x,y,w,h,c))

    return detections, th


# ==============================
# EVENT MEMORY
# ==============================

event_buffer = []


def record_events(frame,detections,depth):

    if len(detections) == 0:
        return

    dvals = []

    for d in detections:

        x,y,w,h,c = d

        cx = x+w//2
        cy = y+h//2

        cy = np.clip(cy,0,depth.shape[0]-1)
        cx = np.clip(cx,0,depth.shape[1]-1)

        dvals.append(depth[cy,cx])

    dval = np.mean(dvals)

    if dval < 0.3:
        event_buffer.append((frame,"near_object"))

    elif dval < 0.6:
        event_buffer.append((frame,"mid_object"))

    else:
        event_buffer.append((frame,"far_object"))


# ==============================
# TIMELINE ENGINE
# ==============================

def build_timeline(events):

    events = sorted(events, key=lambda x: x[0])

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

    print("\n--- Crime Scene Interpretation ---\n")

    for f,e in timeline:

        if e == "near_object":
            print("Actor approached object at frame",f)

        if e == "mid_object":
            print("Actor moved near environment structure at frame",f)

        if e == "far_object":
            print("Actor observing distant object at frame",f)

    print("\nTimeline:\n")

    for t in timeline:
        print(t)


# ==============================
# ANIMATION
# ==============================

fig2 = plt.figure(figsize=(15,6))

ax_scene = fig2.add_subplot(131)
ax_pov = fig2.add_subplot(132)
ax_mask = fig2.add_subplot(133)

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
mask_img = ax_mask.imshow(np.zeros((100,100)), cmap="gray")


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

    depth = estimate_depth(pov)

    detections, mask = detect_silhouettes(pov)

    record_events(frame,detections,depth)

    for x,y,w,h,c in detections:

        cv2.rectangle(pov,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(pov,[c],-1,(255,0,0),2)

    pov_img.set_data(pov)
    mask_img.set_data(mask)

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