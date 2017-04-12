from keras.models import Sequential
from keras.layers import InputLayer, Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


def buildNetwork():
    model = Sequential([
        InputLayer((32, 32, 1)),

        Convolution2D(32, 3, 3, border_mode='valid', init='lecun_uniform'),   # 32x32(x1)  -> 30x30(x32)
        LeakyReLU(alpha=0.3),

        Convolution2D(32, 3, 3, border_mode='valid', init='lecun_uniform'),   # 30x30(x32) -> 28x28(x32)
        LeakyReLU(alpha=0.3),

        Convolution2D(64, 3, 3, border_mode='valid', init='lecun_uniform'),   # 28x28(x64) -> 26x26(x64)
        MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'),    # 26x26(x64) -> 13x13(x64)
        LeakyReLU(alpha=0.3),

        Convolution2D(64, 3, 3, border_mode='valid', init='lecun_uniform'),   # 13x13(x64) -> 11x11(x64)
        LeakyReLU(alpha=0.3),

        Convolution2D(128, 3, 3, border_mode='valid', init='lecun_uniform'),  # 11x11(x64) -> 9x9(x128)
        LeakyReLU(alpha=0.3),

        Convolution2D(64, 9, 9, border_mode='valid', init='lecun_uniform'),   # 9x9(x128)  -> 1x1(x64)
        LeakyReLU(alpha=0.3),

        Flatten(),

        # FC 64 -> 64
        Dense(64, init='lecun_uniform'),
        LeakyReLU(alpha=0.3),

        # FC 64 -> 64
        Dense(64, init='lecun_uniform'),
        LeakyReLU(alpha=0.3),

        # FC 64 -> 1
        Dense(1, init='lecun_uniform', activation='tanh')
    ])

    model.compile(optimizer='adadelta', loss='mse')
    model.load_weights("weights.dat")
    return model


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

    return orig, scale, np.array(grey)


def main(inFile, outFile):
    model = buildNetwork()
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
        subBatchSize = min(32, n - i)
        subBatch = allSamples[i:(i+subBatchSize)]

        networkInput = np.reshape(subBatch, (subBatchSize, 32, 32, 1))
        networkOutput = model.predict(networkInput, subBatchSize, 0)

        for j in range(subBatchSize):
            globalSampleID = i + j
            x, y = samples_xy[globalSampleID]
            heatMap[(y+14):(y+17), (x+11):(x+21)].fill(networkOutput[j, 0])

        i += subBatchSize

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

main('assets/datajobs16.14.jpg', 'heatmap.14.png')
