import matplotlib
matplotlib.use("TkAgg")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge
import cv2
import glob
import open3d as o3d

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

print("Objects loaded:", len(objects))


# ==============================
# LOAD PANORAMAS
# ==============================

pan_files = glob.glob("data/scene_00001/**/panorama.png", recursive=True)

if len(pan_files) == 0:
    raise Exception("No panorama images found")

panorama = cv2.imread(pan_files[0])
panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

H, W, _ = panorama.shape

print("Panorama size:", W, H)


# ==============================
# DRAW SCENE MAP
# ==============================

fig, ax = plt.subplots(figsize=(8,8))

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

ax.set_title("Draw Actor Path")

rectangles = []

for obj in objects:

    cx, cy = obj["center"][:2]
    sx, sy = obj["size"][:2]

    r = Rectangle((cx-sx/2, cy-sy/2), sx, sy,
                  edgecolor="blue",
                  facecolor="none")

    rectangles.append(r)

    ax.add_patch(r)

path = []

def onclick(event):

    if event.xdata is None:
        return

    x = event.xdata
    y = event.ydata

    path.append([x,y])

    ax.plot(x,y,"ro")

    plt.draw()

fig.canvas.mpl_connect("button_press_event", onclick)

plt.show(block=True)

path = np.array(path)

if len(path) < 2:
    print("Draw at least two points")
    exit()


# ==============================
# CAMERA SETTINGS
# ==============================

FOV = 90
VIEW_DIST = 6


def compute_angle(p1,p2):

    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]

    ang = np.degrees(np.arctan2(dy,dx))

    if ang < 0:
        ang += 360

    return ang


# ==============================
# OBJECT VISIBILITY
# ==============================

def visible_objects(actor,angle):

    vis = []

    for obj in objects:

        ox,oy = obj["center"][:2]

        dx = ox-actor[0]
        dy = oy-actor[1]

        dist = np.sqrt(dx**2+dy**2)

        if dist > VIEW_DIST:
            continue

        obj_ang = np.degrees(np.arctan2(dy,dx))

        if obj_ang < 0:
            obj_ang += 360

        diff = abs(obj_ang-angle)

        if diff < FOV/2:

            vis.append((obj,dist,diff))

    return vis


# ==============================
# POV PROJECTION
# ==============================

def project_to_view(dist,angle_offset):

    x = int((angle_offset/FOV+0.5)*W/4)

    size = int(400/(dist+0.5))

    return x,size


# ==============================
# HEATMAP
# ==============================

heatmap = np.zeros((100,100))


def update_heatmap(pos):

    x = int((pos[0]+5)/10*100)
    y = int((pos[1]+5)/10*100)

    if 0<=x<100 and 0<=y<100:
        heatmap[y,x]+=1


# ==============================
# TIMELINE
# ==============================

timeline=[]

def detect_interactions(actor,frame):

    for obj in objects:

        ox,oy=obj["center"][:2]

        dist=np.linalg.norm([ox-actor[0],oy-actor[1]])

        if dist<0.7:

            timeline.append((frame,"interaction",obj["center"]))


# ==============================
# ANIMATION
# ==============================

fig2=plt.figure(figsize=(12,6))

ax_scene=fig2.add_subplot(121)
ax_pov=fig2.add_subplot(122)

ax_scene.set_xlim(-5,5)
ax_scene.set_ylim(-5,5)

scene_rects=[]

for obj in objects:

    cx,cy=obj["center"][:2]
    sx,sy=obj["size"][:2]

    r=Rectangle((cx-sx/2,cy-sy/2),sx,sy,
                edgecolor="blue",
                facecolor="none")

    scene_rects.append(r)

    ax_scene.add_patch(r)

actor=Circle(path[0],0.15,color="red")

ax_scene.add_patch(actor)

cone=Wedge(path[0],3,0,FOV,color="yellow",alpha=0.3)

ax_scene.add_patch(cone)

pov_img=ax_pov.imshow(np.zeros((H,W//4,3),dtype=np.uint8))

ax_pov.set_title("Actor POV")


def update(frame):

    if frame>=len(path)-1:
        return

    p1=path[frame]
    p2=path[frame+1]

    actor.center=p1

    angle=compute_angle(p1,p2)

    cone.set_center(p1)
    cone.theta1=angle-FOV/2
    cone.theta2=angle+FOV/2

    vis=visible_objects(p1,angle)

    pov=np.zeros((H,W//4,3),dtype=np.uint8)

    for obj,dist,off in vis:

        cx,cy=obj["center"][:2]

        dx=cx-p1[0]
        dy=cy-p1[1]

        obj_ang=np.degrees(np.arctan2(dy,dx))

        if obj_ang<0:
            obj_ang+=360

        offset=obj_ang-angle

        x,size=project_to_view(dist,offset)

        y=H//2

        cv2.rectangle(
            pov,
            (x-size,y-size),
            (x+size,y+size),
            (255,0,0),
            2
        )

    pov_img.set_data(pov)

    update_heatmap(p1)

    detect_interactions(p1,frame)

    return actor,pov_img


ani=animation.FuncAnimation(
    fig2,
    update,
    frames=len(path)-1,
    interval=200
)

plt.show()


# ==============================
# HEATMAP VISUALIZATION
# ==============================

plt.figure()

plt.title("Actor Attention Heatmap")

plt.imshow(heatmap,cmap="hot")

plt.colorbar()

plt.show()


# ==============================
# TIMELINE OUTPUT
# ==============================

print("\nCrime Timeline")

for t in timeline[:20]:
    print(t)


# ==============================
# 3D RECONSTRUCTION
# ==============================

print("Launching 3D scene...")

geoms=[]

for obj in objects:

    cx,cy,cz=obj["center"]
    sx,sy,sz=obj["size"]

    box=o3d.geometry.TriangleMesh.create_box(
        width=sx,
        height=sy,
        depth=sz
    )

    box.translate((cx-sx/2,cy-sy/2,cz-sz/2))

    box.paint_uniform_color([0.7,0.7,0.7])

    geoms.append(box)


path3d=[[p[0],p[1],0] for p in path]

lines=[[i,i+1] for i in range(len(path3d)-1)]

ls=o3d.geometry.LineSet()

ls.points=o3d.utility.Vector3dVector(path3d)
ls.lines=o3d.utility.Vector2iVector(lines)

ls.paint_uniform_color([1,0,0])

geoms.append(ls)

o3d.visualization.draw_geometries(geoms)