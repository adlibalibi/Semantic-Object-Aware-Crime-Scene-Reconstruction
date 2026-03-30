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

print("Loaded objects:", len(objects))


# ==============================
# LOAD PANORAMA IMAGES
# ==============================

pan_files = glob.glob("data/scene_00001/**/panorama.png", recursive=True)

panoramas = []

for p in pan_files:

    img = cv2.imread(p)

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    panoramas.append(img)

print("Loaded panoramas:", len(panoramas))

panorama = panoramas[0]

H, W, _ = panorama.shape


# ==============================
# DRAW SCENE + USER PATH
# ==============================

fig, ax = plt.subplots(figsize=(8,8))

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_title("Draw Actor Path")

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

    ax.add_patch(rect)

path = []

def onclick(event):

    if event.xdata is None:
        return

    x = event.xdata
    y = event.ydata

    path.append([x,y])

    ax.plot(x,y,"ro")

    plt.draw()

cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()

path = np.array(path)

print("Recorded path:", len(path))


# ==============================
# POV FROM PANORAMA
# ==============================

FOV = 90

def get_pov(angle):

    center = int((angle / 360) * W)

    width = int((FOV / 360) * W)

    left = center - width//2
    right = center + width//2

    if left < 0:

        view = np.hstack((panorama[:, left:], panorama[:, :right]))

    elif right > W:

        view = np.hstack((panorama[:, left:], panorama[:, :right-W]))

    else:

        view = panorama[:, left:right]

    return view


# ==============================
# ANGLE COMPUTATION
# ==============================

def compute_angle(p1, p2):

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle = np.degrees(np.arctan2(dy, dx))

    if angle < 0:
        angle += 360

    return angle


# ==============================
# VISIBILITY CHECK
# ==============================

def visible_objects(actor_pos, angle):

    visible = []

    for obj in objects:

        ox, oy = obj["center"][:2]

        dx = ox - actor_pos[0]
        dy = oy - actor_pos[1]

        dist = np.sqrt(dx**2 + dy**2)

        if dist > 6:
            continue

        obj_angle = np.degrees(np.arctan2(dy, dx))

        if obj_angle < 0:
            obj_angle += 360

        diff = abs(obj_angle - angle)

        if diff < FOV/2:

            visible.append(obj)

    return visible


# ==============================
# ANIMATION SETUP
# ==============================

fig2 = plt.figure(figsize=(12,6))

ax_scene = fig2.add_subplot(121)
ax_pov = fig2.add_subplot(122)

ax_scene.set_xlim(-5,5)
ax_scene.set_ylim(-5,5)

scene_rects = []

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

    scene_rects.append(r)

    ax_scene.add_patch(r)


actor = Circle((path[0][0],path[0][1]),0.15,color="red")

ax_scene.add_patch(actor)

fov_cone = Wedge(
    path[0],
    3,
    0,
    FOV,
    color="yellow",
    alpha=0.2
)

ax_scene.add_patch(fov_cone)

pov_img = ax_pov.imshow(np.zeros((100,100,3),dtype=np.uint8))

ax_pov.set_title("Actor POV")


# ==============================
# UPDATE FUNCTION
# ==============================

def update(frame):

    if frame >= len(path)-1:
        return

    p1 = path[frame]
    p2 = path[frame+1]

    actor.center = p1

    angle = compute_angle(p1,p2)

    fov_cone.set_center(p1)
    fov_cone.theta1 = angle - FOV/2
    fov_cone.theta2 = angle + FOV/2

    vis = visible_objects(p1,angle)

    for rect in scene_rects:
        rect.set_edgecolor("blue")

    for obj in vis:

        cx, cy = obj["center"][:2]
        sx, sy = obj["size"][:2]

        for rect in scene_rects:

            if abs(rect.get_x() - (cx - sx/2)) < 0.001:
                rect.set_edgecolor("red")

    view = get_pov(angle)

    pov_img.set_data(view)

    return actor, pov_img


ani = animation.FuncAnimation(
    fig2,
    update,
    frames=len(path)-1,
    interval=120,
    blit=False
)

plt.show()


# ==============================
# 3D CRIME SCENE RECONSTRUCTION
# ==============================

print("Launching 3D Reconstruction...")

geometries = []

for obj in objects:

    cx, cy, cz = obj["center"]
    sx, sy, sz = obj["size"]

    box = o3d.geometry.TriangleMesh.create_box(
        width=sx,
        height=sy,
        depth=sz
    )

    box.translate((cx - sx/2, cy - sy/2, cz - sz/2))

    box.paint_uniform_color([0.6,0.6,0.6])

    geometries.append(box)


path_points = []

for p in path:

    path_points.append([p[0],p[1],0])

lines = [[i,i+1] for i in range(len(path_points)-1)]

line_set = o3d.geometry.LineSet()

line_set.points = o3d.utility.Vector3dVector(path_points)
line_set.lines = o3d.utility.Vector2iVector(lines)

line_set.paint_uniform_color([1,0,0])

geometries.append(line_set)

o3d.visualization.draw_geometries(geometries)