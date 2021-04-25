import numpy as np
import warnings
from cv2 import cv2

warnings.filterwarnings("ignore")

image = cv2.imread("img.jpg")

# image = cv2.resize(image, (800, 600))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # Convert to BnW
cv2.imshow('Before', image_gray)    # Show initial image
cv2.waitKey(0)

# Get the values we require of the gray levels.
histg = cv2.calcHist([image_gray], [0], None, [255], [0, 255])
#print(histg)

within = [] # To calculate within class variance
between = [] # To calculate between class variance
d = 0
for i in range(len(histg)):
    x,y = np.split(histg, [i])
    x1 = np.sum(x)/(image.shape[0]*image.shape[1])  # Weight of Class 1
    y1 = np.sum(y)/(image.shape[0]*image.shape[1])

    x2 = np.sum([j*t for j,t in enumerate(x)])/np.sum(x)
    x2 = np.nan_to_num(x2)
    x3 = np.sum([(j-x2)**2*t for j,t in enumerate(x)])/np.sum(x)
    x3 = np.nan_to_num(x3)  # Eliminate Nans

    y2 = np.sum([(j+d)*t for j,t in enumerate(y)])/np.sum(y)
    y3 = np.sum([(j+d-y2)**2*t for j,t in enumerate(y)])/np.sum(y)
    d += 1
    within.append(x1*x3 + y1*y3)
    between.append(x1*y1*(x2-y2)*(x2-y2))


m = np.argmin(within)
n = np.argmax(between)
print("Threshold Value for Within Class: ",m)
print("Threshold Value for Between Class: ",n)      # Both will be same
(thresh, Bin) = cv2.threshold(image_gray, m, 255, cv2.THRESH_BINARY)
cv2.imshow("After",Bin)
cv2.waitKey(0)
