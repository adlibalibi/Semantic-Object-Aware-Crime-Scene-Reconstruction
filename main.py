import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import heapq

# ==============================
# 1. LOAD BBOX DATA
# ==============================

with open("data/scene_00001/bbox_3d.json") as f:
    bboxes = json.load(f)

# Optional object annotations
try:
    with open("data/scene_00001/annotation_3d.json") as f:
        annotations = json.load(f)
except:
    annotations = {}

objects = []

for obj in bboxes:

    centroid = np.array(obj["centroid"]) / 1000.0
    coeffs = np.array(obj["coeffs"]) / 1000.0

    yaw = obj.get("yaw", 0)

    obj_id = obj.get("id", "obj")

    label = annotations.get(obj_id, {}).get("class", "unknown")

    objects.append({
        "id": obj_id,
        "label": label,
        "center": centroid,
        "width": coeffs[0]*2,
        "depth": coeffs[1]*2,
        "height": coeffs[2]*2,
        "yaw": yaw
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

print("Filtered objects:", len(filtered_objects))

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

actor = Actor(1.0,1.0,orientation=0,fov=60)

# ==============================
# 5. OCCUPANCY GRID
# ==============================

grid_resolution = 0.2

grid_width = int(scene_width/grid_resolution)
grid_height = int(scene_height/grid_resolution)

occupancy_grid = np.zeros((grid_width,grid_height))

for obj in filtered_objects:

    rx,ry = obj["center"][:2]
    w,d = obj["width"],obj["depth"]

    left = max(0,int((rx-w/2)/grid_resolution))
    right = min(grid_width-1,int((rx+w/2)/grid_resolution))
    bottom = max(0,int((ry-d/2)/grid_resolution))
    top = min(grid_height-1,int((ry+d/2)/grid_resolution))

    occupancy_grid[left:right+1,bottom:top+1] = 1

# ==============================
# 6. A* PATH PLANNING
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

            if grid[neighbor[0],neighbor[1]]==1:
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

grid_path = astar(occupancy_grid,start,goal)

path=np.array([[p[0]*grid_resolution,p[1]*grid_resolution]
               for p in grid_path])

# ==============================
# 8. VISIBILITY FUNCTIONS
# ==============================

def lines_intersect(p1,p2,q1,q2):

    def ccw(a,b,c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

    return (ccw(p1,q1,q2)!=ccw(p2,q1,q2)) and \
           (ccw(p1,p2,q1)!=ccw(p1,p2,q2))


def line_intersects_rect(p1,p2,rect_center,w,d):

    rx,ry = rect_center[:2]

    left=rx-w/2
    right=rx+w/2
    bottom=ry-d/2
    top=ry+d/2

    edges=[((left,bottom),(right,bottom)),
           ((right,bottom),(right,top)),
           ((right,top),(left,top)),
           ((left,top),(left,bottom))]

    for e in edges:
        if lines_intersect(p1,p2,e[0],e[1]):
            return True

    return False


def is_visible(actor,obj,objects):

    obj_pos = obj["center"][:2]

    direction=np.array([
        np.cos(np.radians(actor.orientation)),
        np.sin(np.radians(actor.orientation))
    ])

    to_obj = obj_pos-actor.position
    dist=np.linalg.norm(to_obj)

    if dist==0:
        return True

    unit_vec=to_obj/dist

    angle=np.degrees(np.arccos(
        np.clip(np.dot(direction,unit_vec),-1,1)
    ))

    if angle>actor.fov/2:
        return False

    for other in objects:

        if other is obj:
            continue

        if line_intersects_rect(actor.position,
                                obj_pos,
                                other["center"],
                                other["width"],
                                other["depth"]):

            other_dist=np.linalg.norm(other["center"][:2]-actor.position)

            if other_dist<dist:
                return False

    return True

# ==============================
# 9. INTERACTION GRAPH
# ==============================

interaction_edges=[]

for i in range(len(filtered_objects)):
    for j in range(i+1,len(filtered_objects)):

        a=filtered_objects[i]["center"][:2]
        b=filtered_objects[j]["center"][:2]

        if np.linalg.norm(a-b)<2.0:
            interaction_edges.append((i,j))

# ==============================
# 10. VISIBILITY LOGGING
# ==============================

visibility_counter={i:0 for i in range(len(filtered_objects))}
event_log=[]

# ==============================
# 11. STATIC SCENE
# ==============================

fig1,ax1=plt.subplots(figsize=(8,8))

for idx,obj in enumerate(filtered_objects):

    x,y=obj["center"][:2]
    w=obj["width"]
    d=obj["depth"]

    rect=Rectangle((x-w/2,y-d/2),w,d,
                   fill=False,edgecolor="black")

    ax1.add_patch(rect)

    ax1.text(x,y,obj["label"],fontsize=7)

ax1.plot(actor.position[0],actor.position[1],"bo")

ax1.set_xlim(0,scene_width)
ax1.set_ylim(0,scene_height)
ax1.set_aspect("equal")
ax1.set_title("Static Scene Layout")

plt.show()

# ==============================
# 12. ANIMATION
# ==============================

fig2,ax2=plt.subplots(figsize=(8,8))

def update(frame):

    ax2.clear()

    actor.position=path[min(frame,len(path)-1)]
    actor.orientation=frame*3

    for i,j in interaction_edges:

        o1=filtered_objects[i]
        o2=filtered_objects[j]

        ax2.plot([o1["center"][0],o2["center"][0]],
                 [o1["center"][1],o2["center"][1]],
                 linestyle=":",color="purple",alpha=0.3)

    for idx,obj in enumerate(filtered_objects):

        visible=is_visible(actor,obj,filtered_objects)

        if visible:
            visibility_counter[idx]+=1
            event_log.append({"frame":frame,
                              "object":obj["label"],
                              "event":"seen"})

        x,y=obj["center"][:2]
        w=obj["width"]
        d=obj["depth"]

        rect=Rectangle((x-w/2,y-d/2),
                       w,d,
                       fill=False,
                       edgecolor="red" if visible else "gray")

        ax2.add_patch(rect)

        ax2.text(x,y,obj["label"],fontsize=6)

        # orientation arrow
        dx=np.cos(obj["yaw"])*0.5
        dy=np.sin(obj["yaw"])*0.5

        ax2.arrow(x,y,dx,dy,color="blue",head_width=0.1)

        # interaction detection
        dist=np.linalg.norm(actor.position-obj["center"][:2])

        if dist<1.5:
            ax2.plot([actor.position[0],x],
                     [actor.position[1],y],
                     color="orange",linewidth=2)

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
    ax2.set_title("Object Aware Simulation")

ani=FuncAnimation(fig2,update,frames=len(path),interval=100)

plt.show()

# ==============================
# 13. VISIBILITY HEATMAP
# ==============================

fig3,ax3=plt.subplots(figsize=(8,8))

max_vis=max(visibility_counter.values())+1e-5

for idx,obj in enumerate(filtered_objects):

    x,y=obj["center"][:2]
    w=obj["width"]
    d=obj["depth"]

    intensity=visibility_counter[idx]/max_vis

    rect=Rectangle((x-w/2,y-d/2),
                   w,d,
                   color=(1,0,0,intensity))

    ax3.add_patch(rect)

ax3.set_xlim(0,scene_width)
ax3.set_ylim(0,scene_height)
ax3.set_aspect("equal")
ax3.set_title("Visibility Heatmap")

plt.show()

# ==============================
# 14. 3D SCENE
# ==============================

def draw_3d_box(ax,center,w,d,h,color):

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
                color=color)

fig4=plt.figure(figsize=(9,9))
ax4=fig4.add_subplot(111,projection="3d")

for idx,obj in enumerate(filtered_objects):

    color="red" if visibility_counter[idx]>0 else "gray"

    draw_3d_box(ax4,
                obj["center"],
                obj["width"],
                obj["depth"],
                obj["height"],
                color)

ax4.set_title("3D Crime Scene Reconstruction")

plt.show()

print("Event log sample:",event_log[:10])