import PIL.Image as Image
import numpy as np
import scipy as sp

import scipy.ndimage as ndimage
import scipy.misc as sp_misc
import matplotlib.pyplot as plt


def get_xy_average(img):
    width = img.shape[1]
    height = img.shape[0]

    if width * height == 0:
        return [], []

    avgX = []
    avgY = []

    for x in range(width):
        avgX.append(np.average(img[:, x]))

    for y in range(height):
        avgY.append(np.average(img[y, :]))

    return avgX, avgY


def bestCuts(array, threshold=0):
    n = len(array)
    if n == 0:
        return []

    maxValue = np.max(array)
    canCutHere = np.greater(array, maxValue - threshold)

    # cluster consecutive occurrences of `True`
    clusters = []
    clusterBegin = -1
    for i in range(n):
        if canCutHere[i]:
            if clusterBegin == -1:  # make sure we are not already inside a cluster
                clusterBegin = i    # begin a cluster
        else:
            if clusterBegin != -1:  # make sure we are inside a cluster
                # if clusterBegin != 0:
                clusters.insert(-1, (clusterBegin, i - clusterBegin))
                clusterBegin = -1   # end the cluster

    if clusterBegin != -1:
        clusters.insert(-1, (clusterBegin, n - clusterBegin))

    return sorted(clusters, key=lambda a: -a[1])


def segment(outCuts, img, depth, xOffset=0, yOffset=0):
    if depth == 0:
        return

    avgX, avgY = get_xy_average(img)
    if len(avgX) == len(avgY) == 0:
        return

    bstX = bestCuts(avgX, 2)
    bstY = bestCuts(avgY, 2)

    if len(bstY) == len(bstX) == 0:
        return

    diff = len(bstY) - len(bstX)
    if diff == 0:
        diff = bstY[0][1] - bstX[0][1]

    if diff < 0:  # vertical cut
        left, width = bstX[0]
        right = left + width
        img1 = img[:, 0:left]
        img2 = img[:, right:]
        outCuts.insert(-1, (xOffset + left, yOffset, xOffset + right, yOffset + img.shape[0]))
        segment(outCuts, img1, depth - 1, xOffset, yOffset)
        segment(outCuts, img2, depth - 1, xOffset + right, yOffset)

    else:  # horizontal cut
        top, height = bstY[0]
        bottom = top + height
        img1 = img[0:top, :]
        img2 = img[bottom:, :]
        outCuts.insert(-1, (xOffset, yOffset + top, xOffset + img.shape[1], yOffset + bottom))
        segment(outCuts, img1, depth - 1, xOffset, yOffset)
        segment(outCuts, img2, depth - 1, xOffset, yOffset + bottom)

orig = Image.open('scaled.8.jpg').convert('L')
heat = Image.open('heatmap.8.png').convert('L')
heat = ndimage.gaussian_filter(heat, 4)

np_orig = np.array(orig)
np_heat = np.array(heat)

h, w = np_orig.shape
textOnly = np.zeros(np_orig.shape)

for y in range(h):
    for x in range(w):
        if np_heat[y, x] > 0:
            textOnly[y, x] = np_orig[y, x]
        else:
            textOnly[y, x] = 255

cuts = []
segment(cuts, textOnly, 16)
for (left, top, right, bottom) in cuts:
    for y in range(top, bottom):
        for x in range(left, right):
            textOnly[y, x] = 200

#plt.subplot(2, 2, 1)
#plt.imshow(np_orig, plt.get_cmap('gray'), clim=(0, 255))
#plt.subplot(2, 2, 2)
#plt.imshow(np_heat, plt.get_cmap('gray'), clim=(0, 255))
plt.subplot(1, 2, 1)
plt.imshow(np_orig, plt.get_cmap('gray'), clim=(0, 255))
plt.subplot(1, 2, 2)
plt.imshow(textOnly, plt.get_cmap('gray'), clim=(0, 255))
plt.show()
