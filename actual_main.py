import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import heapq

# ==============================
# 1. LOAD BBOX DATA
# ==============================

with open("data/scene_00001/bbox_3d.json") as f:
    bboxes = json.load(f)

objects = []

for obj in bboxes:

    centroid = np.array(obj["centroid"]) / 1000.0
    coeffs = np.array(obj["coeffs"]) / 1000.0

    objects.append({
        "center": centroid,
        "width": coeffs[0]*2,
        "depth": coeffs[1]*2,
        "height": coeffs[2]*2
    })

# ==============================
# 2. FILTER OBJECTS
# ==============================

filtered_objects = []

for obj in objects:

    if obj["center"][2] > 2.5:
        continue

    if obj["width"] < 0.2 or obj["depth"] < 0.2:
        continue

    filtered_objects.append(obj)

print("Objects used:", len(filtered_objects))

# ==============================
# 3. NORMALIZE SCENE
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
    def __init__(self,x,y,orientation=0,fov=60):
        self.position = np.array([x,y])
        self.orientation = orientation
        self.fov = fov

actor = Actor(1.0,1.0)

# ==============================
# 5. OCCUPANCY GRID
# ==============================

grid_resolution = 0.2

grid_width = int(scene_width/grid_resolution)
grid_height = int(scene_height/grid_resolution)

grid = np.zeros((grid_width,grid_height))

for obj in filtered_objects:

    x,y = obj["center"][:2]
    w,d = obj["width"],obj["depth"]

    left = int((x-w/2)/grid_resolution)
    right = int((x+w/2)/grid_resolution)

    bottom = int((y-d/2)/grid_resolution)
    top = int((y+d/2)/grid_resolution)

    grid[left:right+1,bottom:top+1] = 1

# ==============================
# 6. A* PATH PLANNER
# ==============================

def heuristic(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

def astar(grid,start,goal):

    neighbors=[(0,1),(1,0),(0,-1),(-1,0)]

    open_set=[]
    heapq.heappush(open_set,(0,start))

    came_from={}
    gscore={start:0}

    while open_set:

        current = heapq.heappop(open_set)[1]

        if current==goal:

            path=[]

            while current in came_from:
                path.append(current)
                current=came_from[current]

            return path[::-1]

        for dx,dy in neighbors:

            neighbor=(current[0]+dx,current[1]+dy)

            if not (0<=neighbor[0]<grid.shape[0] and
                    0<=neighbor[1]<grid.shape[1]):
                continue

            if grid[neighbor]==1:
                continue

            tentative = gscore[current]+1

            if tentative < gscore.get(neighbor,float("inf")):

                came_from[neighbor]=current
                gscore[neighbor]=tentative

                fscore = tentative + heuristic(neighbor,goal)

                heapq.heappush(open_set,(fscore,neighbor))

    return []

# ==============================
# 7. PATH GENERATION
# ==============================

start=(int(1/grid_resolution),int(1/grid_resolution))

goal=(int((scene_width-1)/grid_resolution),
      int((scene_height-1)/grid_resolution))

grid_path = astar(grid,start,goal)

path=np.array([[p[0]*grid_resolution,p[1]*grid_resolution]
               for p in grid_path])

# ==============================
# 8. STATIC SCENE LAYOUT
# ==============================

fig1,ax1=plt.subplots(figsize=(8,8))

for obj in filtered_objects:

    x,y=obj["center"][:2]
    w=obj["width"]
    d=obj["depth"]

    rect=Rectangle((x-w/2,y-d/2),w,d,
                   fill=False,edgecolor="black")

    ax1.add_patch(rect)

ax1.plot(actor.position[0],actor.position[1],"bo")

ax1.set_xlim(0,scene_width)
ax1.set_ylim(0,scene_height)
ax1.set_aspect("equal")
ax1.set_title("Static Scene Layout")

plt.show()

# ==============================
# 9. BASIC SIMULATION
# ==============================

fig2,ax2=plt.subplots(figsize=(8,8))

def update(frame):

    ax2.clear()

    actor.position = path[min(frame,len(path)-1)]
    actor.orientation = frame*3

    for obj in filtered_objects:

        x,y=obj["center"][:2]
        w=obj["width"]
        d=obj["depth"]

        rect=Rectangle((x-w/2,y-d/2),w,d,
                       fill=False,color="gray")

        ax2.add_patch(rect)

    ax2.plot(path[:,0],path[:,1],"g--",alpha=0.3)

    ax2.plot(actor.position[0],actor.position[1],"bo")

    length=2

    left=np.radians(actor.orientation-actor.fov/2)
    right=np.radians(actor.orientation+actor.fov/2)

    ax2.plot([actor.position[0],actor.position[0]+length*np.cos(left)],
             [actor.position[1],actor.position[1]+length*np.sin(left)],
             "b--")

    ax2.plot([actor.position[0],actor.position[0]+length*np.cos(right)],
             [actor.position[1],actor.position[1]+length*np.sin(right)],
             "b--")

    ax2.set_xlim(0,scene_width)
    ax2.set_ylim(0,scene_height)
    ax2.set_aspect("equal")
    ax2.set_title("Basic Actor Simulation")

ani=FuncAnimation(fig2,update,frames=len(path),interval=100)

plt.show()

# ==============================
# 10. 3D CRIME SCENE
# ==============================

from mpl_toolkits.mplot3d import Axes3D

def draw_box(ax,center,w,d,h):

    x,y,z=center

    x-=w/2
    y-=d/2
    z-=h/2

    corners=np.array([
        [x,y,z],
        [x+w,y,z],
        [x+w,y+d,z],
        [x,y+d,z],
        [x,y,z+h],
        [x+w,y,z+h],
        [x+w,y+d,z+h],
        [x,y+d,z+h]
    ])

    edges=[(0,1),(1,2),(2,3),(3,0),
           (4,5),(5,6),(6,7),(7,4),
           (0,4),(1,5),(2,6),(3,7)]

    for e in edges:

        ax.plot([corners[e[0]][0],corners[e[1]][0]],
                [corners[e[0]][1],corners[e[1]][1]],
                [corners[e[0]][2],corners[e[1]][2]],
                color="black")

fig3=plt.figure(figsize=(9,9))
ax3=fig3.add_subplot(111,projection="3d")

for obj in filtered_objects:

    draw_box(ax3,
             obj["center"],
             obj["width"],
             obj["depth"],
             obj["height"])

ax3.set_title("3D Crime Scene Reconstruction")

plt.show()