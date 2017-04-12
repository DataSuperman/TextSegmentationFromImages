from keras.models import Sequential
from keras.layers import InputLayer, Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image, ImageDraw, ImageFont
import os
import random
import re
import numpy as np
import scipy.ndimage as scipy_ndimage
import matplotlib.pyplot as plt


FONT_LIST = [
    "arial.ttf",
    "arialbd.ttf",
    "calibri.ttf",
    "times.ttf",
    "timesbd.ttf",
    "timesi.ttf",
]

FONT_OBJECTS = [
    ImageFont.truetype("text/" + f, s)
    for f in FONT_LIST
    for s in range(10, 30)
]

TEXT_CORPUS = open("text/corpus.txt").read().replace('\n', ' ')

NETWORK_INPUT_W = 32
NETWORK_INPUT_H = 32

# When generating counter-examples (using `genEvilExample()`, we will flip the Y axis of the image. We want the
# resulting image to look like text, but not contain any valid letters. This way, the network will be forced to
# learn what some letters look like, instead of just learning to detect the presence of black ink.
#
# When we flip the Y axis, some letters will still look like valid letters, so if the network has learned to
# recognize valid letters, there is no reason to punish it for recognizing these letters. We will remove a few
# specific letters from the sample text when building counter-examples.
MIRROR_LETTERS_REGEX = re.compile(r"[BbCcDdEHIKlOopqXx038\[\]]")


def getRandomFont(fonts=FONT_OBJECTS):
    return fonts[random.randint(0, len(fonts) - 1)]


def getRandomSnippet(snippetLength=90, corpus=TEXT_CORPUS):
    length = len(corpus)
    snippetLength = min(snippetLength, length)
    snippetStart = random.randint(0, length - 90 - 1)
    snippetStop = snippetStart + snippetLength
    return corpus[snippetStart:snippetStop]


def getRandomForegroundAndBackground():
    a = random.randint(0, 120)
    b = random.randint(a + 30, 255)

    if random.randint(0, 1) == 0:
        a, b = b, a

    return "rgb(%d,%d,%d)" % (a, a, a), "rgb(%d,%d,%d)" % (b, b, b)


def buildNetwork():
    model = Sequential([
        InputLayer((32, 32, 1)),

        Convolution2D(32, 3, 3, border_mode='valid', init='lecun_uniform'),   # 32x32(x1)  -> 30x30(x32)
        LeakyReLU(alpha=0.3),

        Convolution2D(32, 3, 3, border_mode='valid', init='lecun_uniform'),   # 30x30(x32) -> 28x28(x32)
        LeakyReLU(alpha=0.3),

        Convolution2D(64, 3, 3, border_mode='valid', init='lecun_uniform'),   # 28x28(x32) -> 26x26(x64)
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
        Dense(1, init='lecun_uniform', activation='tanh'),
    ])

    model.compile(optimizer='adadelta', loss='mse')
    try:
        model.load_weights("weights.dat")
    except:
        pass

    return model


# A couple of images, used as temporary buffers when generating training examples.
# After an images is painted, `numpy.array(image)` makes a copy, to be sent to the network.
CANVAS_WIDTH = 100
CANVAS_HEIGHT = 80
CANVAS_RECTANGLE = (0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
globalTempImage = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT))
globalTempHeatMap = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT))

# "Painter" objects that implement algorithms for drawing things to an underlying image.
# We are going to use `.rectangle()` and `.text()` to update the `globalTemp*` images above.
globalImagePainter = ImageDraw.Draw(globalTempImage)
globalHeatMapPainter = ImageDraw.Draw(globalTempHeatMap)


def genTextAndHeatMap():
    # generate a random paragraph
    lineCount = random.randint(1, 5)                              # how many lines we want in our paragraph
    lineSpacing = random.randint(0, 10)                           # how many pixels should we leave between lines
    fontObj = getRandomFont()                                     # the font of our paragraph
    textSamples = [getRandomSnippet() for _ in range(lineCount)]  # text fragments to use for each line

    # compute the size of the text (note: text height is probably the same for all samples using the same font)
    sampleSizes = [fontObj.getsize(t) for t in textSamples]       # (width, height) tuples for each line of text
    totalParagraphHeight = (lineCount - 1) * lineSpacing + sum([sizeTuple[1] for sizeTuple in sampleSizes])

    # reset the image and heat map
    fg, bg = getRandomForegroundAndBackground()
    globalImagePainter.rectangle(CANVAS_RECTANGLE, bg)
    globalHeatMapPainter.rectangle(CANVAS_RECTANGLE, "black")

    # paint the text and heat map (to their respective global images)
    y = (CANVAS_HEIGHT / 2) - (totalParagraphHeight / 2)
    for i in range(lineCount):
        # figure out where and what we're going to draw
        textSample = textSamples[i]
        textWidth, textHeight = sampleSizes[i]
        x = random.randint(-textWidth, CANVAS_WIDTH)     # text can partially or completely fall outside the canvas

        # draw the text
        globalImagePainter.text((x, y), textSample, font=fontObj, fill=fg)

        # Draw a rectangle in the heat map, but make it slightly smaller (remove 2-3 pixels on each side).
        # This is because we will blur the heat map a bit to make the transition less abrupt
        # and we also want the blurred edges to not overlap with the rectangles above and below.
        heatRect = (x + 1, y + 3, x + textWidth - 1, y + textHeight - 3)
        globalHeatMapPainter.rectangle(heatRect, "white")

        # update the "y" component to prepare for the next line
        y = y + textHeight + lineSpacing

    # blur the heat map
    blurredHeatMap = scipy_ndimage.gaussian_filter(globalTempHeatMap, 2)

    # clone the (global) generated image and heat map into a pair of numpy 2D arrays
    return np.array(globalTempImage), (np.array(blurredHeatMap, dtype=np.float32)/255)


def genEvilExample():
    # generate a random paragraph
    lineCount = random.randint(1, 5)                              # how many lines we want in our paragraph
    lineSpacing = random.randint(0, 10)                           # how many pixels should we leave between lines
    fontObj = getRandomFont()                                     # the font of our paragraph
    textSamples = [getRandomSnippet() for _ in range(lineCount)]  # text fragments to use for each line
    textSamples = [MIRROR_LETTERS_REGEX.sub('', t) for t in textSamples]  # see comment on `MIRROR_LETTERS_REGEX`

    # compute the size of the text (note: text height is probably the same for all samples using the same font)
    sampleSizes = [fontObj.getsize(t) for t in textSamples]       # (width, height) tuples for each line of text
    totalParagraphHeight = (lineCount - 1) * lineSpacing + sum([sizeTuple[1] for sizeTuple in sampleSizes])

    # reset the image and heat map
    fg, bg = getRandomForegroundAndBackground()
    globalImagePainter.rectangle(CANVAS_RECTANGLE, bg)

    # paint the text and heat map (to their respective global images)
    y = (CANVAS_HEIGHT / 2) - (totalParagraphHeight / 2)
    for i in range(lineCount):
        # figure out where and what we're going to draw
        textSample = textSamples[i]
        textWidth, textHeight = sampleSizes[i]
        x = random.randint(-textWidth, CANVAS_WIDTH)     # text can partially or completely fall outside the canvas

        # draw the text
        globalImagePainter.text((x, y), textSample, font=fontObj, fill=fg)

        # update the "y" component to prepare for the next line
        y = y + textHeight + lineSpacing

    return np.array(globalTempImage)[::-1, :]  # "[::-1, :]" is used to flip the Y axis


def genNoEasyEdges():
    fg, bg = getRandomForegroundAndBackground()
    globalImagePainter.rectangle(CANVAS_RECTANGLE, bg)

    for _ in range(3):
        x1 = random.randint(0, CANVAS_WIDTH)
        x2 = random.randint(0, CANVAS_WIDTH)
        y1 = random.randint(0, CANVAS_HEIGHT)
        y2 = random.randint(0, CANVAS_HEIGHT)

        shape = random.randint(0, 4)
        if shape == 0:  # line
            globalImagePainter.line((x1, x2, y1, y2), fg, random.randint(1, 32))
        elif shape == 1:  # rectangle
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            globalImagePainter.rectangle((x1, x2, y1, y2), fg)
        elif shape == 2:  # ellipse
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            globalImagePainter.ellipse((x1, x2, y1, y2), fg)
        elif shape == 3:  # triangle
            x3 = random.randint(0, CANVAS_WIDTH)
            y3 = random.randint(0, CANVAS_HEIGHT)
            globalImagePainter.polygon([(x1, y1), (x2, y2), (x3, y3)], fg)
        else:             # some vertical and horizontal lines
            globalImagePainter.line((x1, 0, x1, CANVAS_HEIGHT), fg)
            globalImagePainter.line((x2, 0, x2, CANVAS_HEIGHT), fg)
            globalImagePainter.line((0, y1, CANVAS_WIDTH, y1), fg)
            globalImagePainter.line((0, y2, CANVAS_WIDTH, y2), fg)

    image = np.array(globalTempImage, dtype=np.float32)
    image = scipy_ndimage.gaussian_filter(image, random.random())
    image += 0.5 * np.random.poisson(image) * random.random()
    return image

lastNormalCanvas = None
lastEvilCanvas = None
lastNoEasyCanvas = None


def genTrainingSet():
    global lastNormalCanvas
    global lastEvilCanvas
    global lastNoEasyCanvas

    X = []
    Y = []

    canvas, heatMap = genTextAndHeatMap()
    lastNormalCanvas = canvas
    for x in range(0, CANVAS_WIDTH - NETWORK_INPUT_W, 10):
        for y in range(0, CANVAS_HEIGHT - NETWORK_INPUT_H, 2):
            X.append(canvas[y:(y+NETWORK_INPUT_H), x:(x+NETWORK_INPUT_W)])
            Y.append(heatMap[round(y + NETWORK_INPUT_H/2), round(x + NETWORK_INPUT_W/2)] * 2 - 1)

    canvas = genEvilExample()
    lastEvilCanvas = canvas
    for x in range(0, CANVAS_WIDTH - NETWORK_INPUT_W, 10):
        for y in range(0, CANVAS_HEIGHT - NETWORK_INPUT_H, 2):
            X.append(canvas[y:(y+NETWORK_INPUT_H), x:(x+NETWORK_INPUT_W)])
            Y.append(-1)

    canvas = genNoEasyEdges()
    lastNoEasyCanvas = canvas
    for x in range(0, CANVAS_WIDTH - NETWORK_INPUT_W, 10):
        for y in range(0, CANVAS_HEIGHT - NETWORK_INPUT_H, 2):
            X.append(canvas[y:(y+NETWORK_INPUT_H), x:(x+NETWORK_INPUT_W)])
            Y.append(-1)

    n = len(X)
    return np.reshape(X, (n, 32, 32, 1)), np.reshape(Y, (n, 1))


model = buildNetwork()
globalTrainingHistory = []

i = 0
while True:
    X, Y = genTrainingSet()

    loss = model.fit(X, Y, nb_epoch=1, batch_size=7, verbose=0).history['loss'][-1]

    globalTrainingHistory.append(loss)
    if len(globalTrainingHistory) > 50:
        globalTrainingHistory.pop(0)

    i += 1
    if i >= 25:
        try:
            os.unlink("weights.dat")
        except:
            pass
        os.rename("weights.dat", "weights.dat")
        model.save_weights("weights.dat")
        i = 0

    if True:
        plt.ion()
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.imshow(lastNormalCanvas, plt.get_cmap('gray'), clim=(0, 255))
        plt.subplot(2, 2, 2)
        plt.imshow(lastEvilCanvas, plt.get_cmap('gray'), clim=(0, 255))
        plt.subplot(2, 2, 3)
        plt.imshow(lastNoEasyCanvas, plt.get_cmap('gray'), clim=(0, 255))
        plt.subplot(2, 2, 4)
        plt.plot(globalTrainingHistory, color='red')
        plt.pause(.001)
        plt.show()

