import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. LOAD MIDAS MODEL
# ==============================

model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "MiDaS_small":
    transform = midas_transforms.small_transform
else:
    transform = midas_transforms.default_transform


# ==============================
# 2. LOAD IMAGE
# ==============================

img = cv2.imread("data\scene_00001\906322\panorama.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)


# ==============================
# 3. DEPTH ESTIMATION
# ==============================

with torch.no_grad():

    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()


# ==============================
# 4. NORMALIZE DEPTH
# ==============================

depth_norm = cv2.normalize(
    depth_map,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# ==============================
# 5. EDGE DETECTION
# ==============================

edges = cv2.Canny(depth_norm, 50, 150)


# ==============================
# 6. CONTOUR EXTRACTION
# ==============================

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# ==============================
# 7. DRAW CONTOURS
# ==============================

depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

contour_img = depth_color.copy()

cv2.drawContours(
    contour_img,
    contours,
    -1,
    (0,255,0),
    2
)


# ==============================
# 8. VISUALIZATION
# ==============================

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Depth Map")
plt.imshow(depth_norm, cmap="inferno")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Depth + Contours")
plt.imshow(contour_img)
plt.axis("off")
plt.show()