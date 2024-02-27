# Get / Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Path to the image file
image_path = "C:\\Users\\Dell\\Desktop\\AI\\Machine Learning\\Cam Scanner\\OIP.jpg"

# Read image from path
img = cv2.imread(image_path)

# Check the shape of the image
print(img.shape)

#image resize

img = cv2.resize(img, (800, 800))
# BGR format : BGR -> RGB
print(img.shape)
plt.imshow(img)
plt.show()


# Remove the Noise
# Edge Detection
# Contour Extraction
# Best Contour Selection
# Project to the Screen

# 1 Remove the Noise
# Image Blurring

original = img.copy()
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap = "binary")
plt.show()

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
plt.imshow(blurred, cmap = "binary")
plt.show()

# Regeneration of image

regen = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

plt.imshow(original)
plt.show()

plt.imshow(regen)
plt.show()
print(regen.shape)

# 2 Edge Detection

edge = cv2.Canny(blurred, 0,50)
orig_edge = edge.copy()

plt.imshow(orig_edge)
plt.title("Edge Detection")
plt.show()

# 3 Contours Extraction

contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(len(contours))

contours = sorted( contours, reverse = True, key = cv2.contourArea )

# 4 Select the Best Contour Region

for c in contours:
    p = cv2.arcLength(c, True)

    approx = cv2.approxPolyDP(c, 0.01*p, True)

    if len(approx) == 4:
        target = approx
        break
print(target.shape)

# Reorder Target Contour

def reorder(h):
    h = h.reshape((4, 2))
    print(h)

    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(axis=0)  # Calculate along columns (axis=0)
    hnew[3] = h[np.argmax(add)]
    hnew[1] = h[np.argmax(add)]

    diff = np.diff(h, axis=0)  # Calculate along columns (axis=0)
    hnew[0] = h[np.argmax(diff)]
    hnew[2] = h[np.argmax(diff)]

    return hnew

reordered = reorder(target)
print("_-_-_-_-_-_-_-_-_-_")
print(reordered)

# 5 Project to a fixed screen

input_representation = reordered.astype(np.float32)  # Ensure input is float32
output_map = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])

M = cv2.getPerspectiveTransform(input_representation, output_map)

ans = cv2.warpPerspective(original, M, (800, 800))

plt.imshow(ans)
plt.show()
