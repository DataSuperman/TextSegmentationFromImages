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


class TextDetectorNetwork:
    """
        A neural network for detecting the presence of text in an image.
        We assume that the text is perfectly horizontal and it uses the Latin alphabet.
        We allow for varying levels of contrast in the input image.

        The structure of the network is as follows:
            - 32x32 gray scale input, not normalized
            - 3x3 convolution with 32 kernels, leaky ReLU; output is 30x30 (depth is 32)
            - 3x3 convolution with 32 kernels, leaky ReLU; output is 28x28 (depth is 32)
            - 3x3 convolution with 64 kernels, 2x2 max-pooling, leaky ReLU; output is 13x13 (depth is 64)
            - 3x3 convolution with 64 kernels, leaky ReLU; output is 11x11 (depth is 64)
            - 3x3 convolution with 128 kernels, leaky ReLU; output is 9x9 (depth is 128)
            - 9x9 convolution with 64 kernels, leaky ReLU; output is 1x1 (depth is 64)
            - fully connected layer (64 inputs to 64 outputs), leaky ReLU
            - fully connected layer (64 inputs to 64 outputs), leaky ReLU
            - fully connected layer (64 inputs, 1 output), tanh

        The (single) output neuron returns the value "1" when it detects text in the center of the 32x32 input.
        The output is "-1" if the neuron is confident there is no text in the input.

        By sliding the network over a larger image, it can be used to build an activation map. The map will contain
        a large value if there is some text at the corresponding position in the input.

    """

    def __init__(self):
        self.model = Sequential([
            InputLayer((32, 32, 1)),

            # 32x32(x1) -> 30x30(x32)
            Convolution2D(32, 3, 3, border_mode='valid', init='lecun_uniform'),
            LeakyReLU(alpha=0.3),

            # 30x30(x32) -> 28x28(x32)
            Convolution2D(32, 3, 3, border_mode='valid', init='lecun_uniform'),
            LeakyReLU(alpha=0.3),

            # 28x28(x32) -> 26x26(x64)
            Convolution2D(64, 3, 3, border_mode='valid', init='lecun_uniform'),

            # 26x26(x64) -> 13x13(x64)
            MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'),
            LeakyReLU(alpha=0.3),

            # 13x13(x64) -> 11x11(x64)
            Convolution2D(64, 3, 3, border_mode='valid', init='lecun_uniform'),
            LeakyReLU(alpha=0.3),

            # 11x11(x64) -> 9x9(x128)
            Convolution2D(128, 3, 3, border_mode='valid', init='lecun_uniform'),
            LeakyReLU(alpha=0.3),

            # 9x9(x128) -> 1x1(x64)
            Convolution2D(64, 9, 9, border_mode='valid', init='lecun_uniform'),
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

        self.model.compile(optimizer='adadelta', loss='mse')

    def loadWeights(self, filename):
        self.model.load_weights(filename)

    def saveWeights(self, filename):
        self.model.save_weights(filename)

    def run(self, batch):
        batchSize = int(batch.size / 1024)
        inputBuffer = np.reshape(batch, (batchSize, 32, 32, 1))
        outputBuffer = self.model.predict(inputBuffer, batchSize)
        return np.reshape(outputBuffer, batchSize)


class TextDetectorTrainer:
    """
        A class that generates random training examples for the text detector neural net.
        Some examples are pictures of text, some examples are adversarial (to prevent the network from
        learning a simpler function than what we want).
    """

    def __init__(self, model, corpus=None, fonts=None):
        self.net = model

        if corpus is not None:
            self.TEXT_CORPUS = corpus
        else:
            self.TEXT_CORPUS = open("text/corpus.txt").read().replace('\n', ' ')

        if fonts is not None:
            self.FONT_LIST = fonts
        else:
            self.FONT_LIST = [
                "text/arial.ttf",
                "text/arialbd.ttf",
                "text/calibri.ttf",
                "text/times.ttf",
                "text/timesbd.ttf",
                "text/timesi.ttf",
            ]

        self.FONT_OBJECTS = [
            ImageFont.truetype(font, size)
            for font in self.FONT_LIST
            for size in range(10, 30)
        ]

        # When generating adversarial examples (using `genEvilExample()`), we will flip the Y axis of the image.
        # We want the resulting image to look like text, but not contain any valid letters. This way, the network
        # will be forced to learn what some letters look like, instead of just learning to detect the presence of
        # black ink or noise.
        #
        # When we flip the Y axis, some letters will still look like valid letters, so if the network has learned
        # to recognize valid letters, there is no reason to punish it for recognizing these letters. We will remove
        # a few specific letters from the sample text when building counter-examples.
        self.MIRROR_LETTERS_REGEX = re.compile(r"[BbCcDdEHIKlOopqXx038\[\]]")

        # A couple of images, used as temporary buffers when generating training examples.
        # After an images is painted, `numpy.array(image)` makes a copy, to be sent to the network.
        self.CANVAS_WIDTH = 100
        self.CANVAS_HEIGHT = 80
        self.CANVAS_RECTANGLE = (0, 0, self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
        self.tempCanvas = Image.new("L", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT))
        self.tempHeatMap = Image.new("L", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT))

        # "Painter" objects encapsulate the implementation of algorithms for drawing to an image.
        # With these objects, we will use `.rectangle()` and `.text()` to cause updates to the `temp*` images above.
        self.tempCanvasPainter = ImageDraw.Draw(self.tempCanvas)
        self.tempHeatMapPainter = ImageDraw.Draw(self.tempHeatMap)

    def getRandomFont(self):
        fonts = self.FONT_OBJECTS
        return fonts[random.randint(0, len(fonts) - 1)]

    def getRandomSnippet(self, snippetLength=30):
        corpus = self.TEXT_CORPUS
        corpusLength = len(corpus)
        snippetLength = min(snippetLength, corpusLength)
        snippetStart = random.randint(0, corpusLength - snippetLength - 1)
        snippetStop = snippetStart + snippetLength
        return corpus[snippetStart:snippetStop]

    @staticmethod
    def getRandomForegroundAndBackground():
        a = random.randint(0, 120)
        b = random.randint(a + 30, 255)

        if random.randint(0, 1) == 0:
            a, b = b, a

        return "rgb(%d,%d,%d)" % (a, a, a), "rgb(%d,%d,%d)" % (b, b, b)

    def genTextAndHeatMap(self):
        # generate a random paragraph
        lineCount = random.randint(1, 5)                                   # how many lines we want in our paragraph
        lineSpacing = random.randint(0, 10)                                # how many pixels to leave between lines
        fontObj = self.getRandomFont()                                     # the font of our paragraph
        textSamples = [self.getRandomSnippet() for _ in range(lineCount)]  # text fragments to use for each line

        # compute the size of the text (note: text height is probably the same for all samples using the same font)
        sampleSizes = [fontObj.getsize(t) for t in textSamples]       # (width, height) tuples for each line of text
        totalParagraphHeight = (lineCount - 1) * lineSpacing + sum([sizeTuple[1] for sizeTuple in sampleSizes])

        # reset the image and heat map
        fg, bg = self.getRandomForegroundAndBackground()
        self.tempCanvasPainter.rectangle(self.CANVAS_RECTANGLE, bg)
        self.tempHeatMapPainter.rectangle(self.CANVAS_RECTANGLE, "black")

        # paint the text and heat map (to their respective images)
        y = (self.CANVAS_HEIGHT / 2) - (totalParagraphHeight / 2)
        for i in range(lineCount):
            # figure out where and what we're going to draw
            textSample = textSamples[i]
            textWidth, textHeight = sampleSizes[i]
            x = random.randint(-textWidth, self.CANVAS_WIDTH)  # text may (intentionally) fall outside the canvas

            # draw the text
            self.tempCanvasPainter.text((x, y), textSample, font=fontObj, fill=fg)

            # Draw a rectangle in the heat map, but make it slightly smaller (remove 2-3 pixels on each side).
            # This is because we will blur the heat map a bit to make the transition less abrupt
            # and we also want the blurred edges to not overlap with the rectangles above and below.
            heatRect = (x + 2, y + 3, x + textWidth - 2, y + textHeight - 3)
            self.tempHeatMapPainter.rectangle(heatRect, "white")

            # update the "y" component to prepare for the next line
            y = y + textHeight + lineSpacing

        # blur the heat map
        blurredHeatMap = scipy_ndimage.gaussian_filter(self.tempHeatMap, 4)

        # clone the generated image and heat map into a pair of numpy 2D arrays
        canvas = np.array(self.tempCanvas, dtype=np.float32)
        heatMap = np.array(blurredHeatMap, dtype=np.float32)/255

        # add some random noise and blur to the canvas
        canvas = scipy_ndimage.gaussian_filter(canvas, random.random())
        canvas += 0.5 * np.random.poisson(canvas) * random.random()

        return canvas, heatMap

    def genEvilExample(self):
        # generate a random paragraph
        lineCount = random.randint(1, 5)                                   # how many lines we want in our paragraph
        lineSpacing = random.randint(0, 10)                                # how many pixels to leave between lines
        fontObj = self.getRandomFont()                                     # the font of our paragraph
        textSamples = [self.getRandomSnippet() for _ in range(lineCount)]  # text fragments to use for each line
        textSamples = [self.MIRROR_LETTERS_REGEX.sub('', ts) for ts in textSamples]  # see `MIRROR_LETTERS_REGEX`

        # compute the size of the text (note: text height is probably the same for all samples using the same font)
        sampleSizes = [fontObj.getsize(t) for t in textSamples]       # (width, height) tuples for each line of text
        totalParagraphHeight = (lineCount - 1) * lineSpacing + sum([sizeTuple[1] for sizeTuple in sampleSizes])

        # reset the image and heat map
        fg, bg = self.getRandomForegroundAndBackground()
        self.tempCanvasPainter.rectangle(self.CANVAS_RECTANGLE, bg)

        # paint the text and heat map (to their respective images)
        y = (self.CANVAS_HEIGHT / 2) - (totalParagraphHeight / 2)
        for i in range(lineCount):
            # figure out where and what we're going to draw
            textSample = textSamples[i]
            textWidth, textHeight = sampleSizes[i]
            x = random.randint(-textWidth, self.CANVAS_WIDTH)  # text may (intentionally) fall outside the canvas

            # draw the text
            self.tempCanvasPainter.text((x, y), textSample, font=fontObj, fill=fg)

            # update the "y" component to prepare for the next line
            y = y + textHeight + lineSpacing

        # clone the generated image a pair of numpy array; add some noise and blur
        canvas = np.array(self.tempCanvas, dtype=np.float32)
        canvas = scipy_ndimage.gaussian_filter(canvas, random.random())
        canvas += 0.5 * np.random.poisson(canvas) * random.random()

        return canvas[::-1, :]  # "[::-1, :]" is used to flip the Y axis

    def genNoEasyEdges(self):
        fg, bg = self.getRandomForegroundAndBackground()
        self.tempCanvasPainter.rectangle(self.CANVAS_RECTANGLE, bg)

        for _ in range(3):
            x1 = random.randint(0, self.CANVAS_WIDTH)
            x2 = random.randint(0, self.CANVAS_WIDTH)
            y1 = random.randint(0, self.CANVAS_HEIGHT)
            y2 = random.randint(0, self.CANVAS_HEIGHT)

            shape = random.randint(0, 4)
            if shape == 0:  # line
                self.tempCanvasPainter.line((x1, x2, y1, y2), fg, random.randint(1, 32))
            elif shape == 1:  # rectangle
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                self.tempCanvasPainter.rectangle((x1, x2, y1, y2), fg)
            elif shape == 2:  # ellipse
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                self.tempCanvasPainter.ellipse((x1, x2, y1, y2), fg)
            elif shape == 3:  # triangle
                x3 = random.randint(0, self.CANVAS_WIDTH)
                y3 = random.randint(0, self.CANVAS_HEIGHT)
                self.tempCanvasPainter.polygon([(x1, y1), (x2, y2), (x3, y3)], fg)
            else:             # some vertical and horizontal lines
                self.tempCanvasPainter.line((x1, 0, x1, self.CANVAS_HEIGHT), fg)
                self.tempCanvasPainter.line((x2, 0, x2, self.CANVAS_HEIGHT), fg)
                self.tempCanvasPainter.line((0, y1, self.CANVAS_WIDTH, y1), fg)
                self.tempCanvasPainter.line((0, y2, self.CANVAS_WIDTH, y2), fg)

        image = np.array(self.tempCanvas, dtype=np.float32)
        image = scipy_ndimage.gaussian_filter(image, random.random())
        image += 0.5 * np.random.poisson(image) * random.random()
        return image

    lastNormalCanvas = None
    lastEvilCanvas = None
    lastNoEasyCanvas = None
    trainingHistory = []

    def genTrainingSet(self):
        X = []
        Y = []

        canvas, heatMap = self.genTextAndHeatMap()
        self.lastNormalCanvas = canvas
        for x in range(0, self.CANVAS_WIDTH - 32, 10):
            for y in range(0, self.CANVAS_HEIGHT - 32, 2):
                X.append(canvas[y:(y+32), x:(x+32)])
                Y.append(heatMap[round(y + 16), round(x + 16)] * 2 - 1)

        canvas = self.genEvilExample()
        self.lastEvilCanvas = canvas
        for x in range(0, self.CANVAS_WIDTH - 32, 10):
            for y in range(0, self.CANVAS_HEIGHT - 32, 2):
                X.append(canvas[y:(y+32), x:(x+32)])
                Y.append(-1)

        canvas = self.genNoEasyEdges()
        self.lastNoEasyCanvas = canvas
        for x in range(0, self.CANVAS_WIDTH - 32, 10):
            for y in range(0, self.CANVAS_HEIGHT - 32, 2):
                X.append(canvas[y:(y+32), x:(x+32)])
                Y.append(-1)

        n = len(X)
        return np.reshape(X, (n, 32, 32, 1)), np.reshape(Y, (n, 1))

    def trainingRound(self):
        X, Y = self.genTrainingSet()
        hist = self.net.model.fit(X, Y, nb_epoch=1, batch_size=7, verbose=0)  # 7 happens to divide len(X)
        loss = hist.history['loss'][-1]
        return loss

    def trainingLoop(self, n, interactive=True):
        for i in range(n):
            loss = self.trainingRound()
            self.trainingHistory.append(loss)
            if len(self.trainingHistory) > 50:
                self.trainingHistory.pop(0)

            if i % 25 == 0:
                try:
                    os.unlink("weights.bak")
                except:
                    pass
                os.rename("weights.dat", "weights.bak")
                self.net.saveWeights("weights.dat")

            if interactive:
                plt.ion()
                plt.clf()
                plt.subplot(2, 2, 1)
                plt.imshow(self.lastNormalCanvas, plt.get_cmap('gray'), clim=(0, 255))
                plt.subplot(2, 2, 2)
                plt.imshow(self.lastEvilCanvas, plt.get_cmap('gray'), clim=(0, 255))
                plt.subplot(2, 2, 3)
                plt.imshow(self.lastNoEasyCanvas, plt.get_cmap('gray'), clim=(0, 255))
                plt.subplot(2, 2, 4)
                plt.plot(self.trainingHistory, color='red')
                plt.pause(.001)
                plt.show()
