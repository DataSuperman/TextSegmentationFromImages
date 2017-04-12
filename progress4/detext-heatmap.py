from progress4.detext import TextDetectorNetwork
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

MAX_WH = 1000, 1600


def loadImage(path):
    orig = Image.open(path)
    grey = orig.convert('L')

    w, h = orig.size
    scale = 1

    if w > MAX_WH[0]:
        scale = MAX_WH[0]/w

    if h*scale > MAX_WH[1]:
        scale *= MAX_WH[1] / h

    w = int(w * scale)
    h = int(h * scale)
    grey = grey.resize((w, h), Image.LINEAR)

    return orig, scale, np.array(grey, dtype=np.float32)


def main(inFile, outFile):
    net = TextDetectorNetwork()
    net.loadWeights("weights.dat")

    _, scale, grey = loadImage(inFile)

    h, w = grey.shape
    heatMap = np.zeros((h, w), dtype=np.float32)
    heatMap.fill(0.2)

    allSamples = []
    samples_xy = []
    for y in range(0, h - 32, 3):
        for x in range(0, w - 32, 10):
            allSamples.append(grey[y:(y+32), x:(x+32)])
            samples_xy.append((x, y))

    i = 0
    n = len(allSamples)
    while i < n:
        batchSize = min(32, n - i)
        batch = allSamples[i:(i+batchSize)]

        networkOutput = net.run(np.array(batch))

        for j in range(batchSize):
            globalSampleID = i + j
            x, y = samples_xy[globalSampleID]
            heatMap[(y+14):(y+17), (x+11):(x+21)].fill(networkOutput[j])

        i += batchSize

        if True:
            plt.ion()
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(grey, plt.get_cmap('gray'), clim=(0, 255))
            plt.subplot(1, 2, 2)
            plt.imshow(heatMap, plt.get_cmap('gray'), clim=(0, 1))
            plt.pause(.001)
            plt.show()

    scipy.misc.imsave(outFile, heatMap)

main('calibration/font-size.png', 'calibration/heat-map.png')
