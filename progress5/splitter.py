from PIL import Image, ImageDraw
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class SubdivisionNode:
    def __init__(self, image, imageRange=None):
        if imageRange is not None:
            self.x1, self.y1, self.x2, self.y2 = imageRange
            self.width = self.x2 - self.x1
            self.height = self.y2 - self.y1
        else:
            self.x1 = self.y1 = 0
            self.height, self.width = self.y2, self.x2 = image.shape

        self.originalImage = image
        self.viewIntoImage = image[self.y1:self.y2, self.x1:self.x2]
        self.minPixel = np.min(self.viewIntoImage)
        self.maxPixel = np.max(self.viewIntoImage)
        self.minPixelByX = np.array([np.min(self.viewIntoImage[:, x]) for x in range(self.width)])
        self.minPixelByY = np.array([np.min(self.viewIntoImage[y, :]) for y in range(self.height)])
        self.isValid = (self.maxPixel - self.minPixel) > 20  # if there's enough contrast in the image
        self.children = []

        if self.isValid:
            removeLeft = 0
            while removeLeft < self.width and self.minPixelByX[removeLeft] > self.maxPixel - 20:
                removeLeft += 1

            removeRight = 0
            while removeRight < self.width - removeLeft and self.minPixelByX[self.width - removeRight - 1] > self.maxPixel - 20:
                removeRight += 1

            removeTop = 0
            while removeTop < self.height and self.minPixelByY[removeTop] > self.maxPixel - 20:
                removeTop += 1

            removeBottom = 0
            while removeBottom < self.height - removeTop and self.minPixelByY[self.height - removeBottom - 1] > self.maxPixel - 20:
                removeBottom += 1

            self.x1 += removeLeft
            self.y1 += removeTop
            self.x2 -= removeRight
            self.y2 -= removeBottom
            self.width = self.x2 - self.x1
            self.height = self.y2 - self.y1
            if self.width <= 0 or self.height <= 0:
                self.isValid = False
                return

            self.viewIntoImage = image[self.y1:self.y2, self.x1:self.x2]
            self.minPixel = np.min(self.viewIntoImage)
            self.maxPixel = np.max(self.viewIntoImage)
            self.minPixelByX = np.array([np.min(self.viewIntoImage[:, x]) for x in range(self.width)])
            self.minPixelByY = np.array([np.min(self.viewIntoImage[y, :]) for y in range(self.height)])

    def subdivideAlongX(self, threshold, minCutWidth):
        xCuts = self.buildCluster(self.minPixelByX > self.maxPixel - threshold, minCutWidth)

        if len(xCuts) == 0:
            return [(0, self.width)]

        resultingSlices = []
        lastX = 0
        for cutBegin, cutWidth in xCuts:
            cutEnd = cutBegin + cutWidth
            if cutBegin != lastX:
                resultingSlices.append((lastX, cutBegin))
            lastX = cutEnd

        if lastX < self.width:
            resultingSlices.append((lastX, self.width))

        return resultingSlices

    def subdivideAlongY(self, threshold, minCutWidth):
        yCuts = self.buildCluster(self.minPixelByY > self.maxPixel - threshold, minCutWidth)

        if len(yCuts) == 0:
            return [(0, self.height)]

        resultingSlices = []
        lastY = 0
        for cutBegin, cutWidth in yCuts:
            cutEnd = cutBegin + cutWidth
            if cutBegin != lastY:
                resultingSlices.append((lastY, cutBegin))
            lastY = cutEnd

        if lastY < self.height:
            resultingSlices.append((lastY, self.height))

        return resultingSlices

    def subdivide(self, threshold, minXCutWidth, minYCutWidth, minSliceWidth, minSliceHeight):
        xSlices = self.subdivideAlongX(threshold, minXCutWidth)
        ySlices = self.subdivideAlongY(threshold, minYCutWidth)
        xSlices = [s for s in xSlices if s[1] >= minSliceWidth]
        ySlices = [s for s in ySlices if s[1] >= minSliceHeight]
        return [(x1, y1, x2, y2) for x1, x2 in xSlices for y1, y2 in ySlices]

    def join(self, node):
        x1 = min(self.x1, node.x1)
        y1 = min(self.y1, node.y1)
        x2 = max(self.x2, node.x2)
        y2 = max(self.y2, node.y2)
        return SubdivisionNode(self.originalImage, (x1, y1, x2, y2))

    @staticmethod
    def buildCluster(candidates, minCutWidth):
        foundClusters = []

        clusterStart = None
        for index, value in enumerate(candidates):
            if value:
                if clusterStart is None:
                    clusterStart = index
            else:
                if clusterStart is not None:
                    clusterLength = index - clusterStart
                    if clusterLength >= minCutWidth:
                        foundClusters.append((clusterStart, clusterLength))
                clusterStart = None

        if clusterStart is not None:
            clusterLength = len(candidates) - clusterStart
            if clusterLength >= minCutWidth:
                foundClusters.append((clusterStart, clusterLength))

        return foundClusters


def subdivideForever(targetNode, threshold):
    slices = targetNode.subdivide(threshold, 1, 1, 2, 2)

    if len(slices) == 1:
        if slices[0] == (0, 0, targetNode.width, targetNode.height):
            return targetNode

    for x1, y1, x2, y2 in slices:
        globalSlice = (x1 + targetNode.x1, y1 + targetNode.y1, x2 + targetNode.x1, y2 + targetNode.y1)
        child = SubdivisionNode(targetNode.originalImage, globalSlice)

        if child.isValid:
            targetNode.children.append(child)
            subdivideForever(child, threshold)


def getLeaves(targetNode):
    if len(targetNode.children) == 0:
        return [targetNode]

    result = []

    for c in targetNode.children:
        result += getLeaves(c)

    return result


class GraphNode:
    def __init__(self, leaf: SubdivisionNode):
        self.id = None
        self.originalBox = leaf

    def similarityScore(self, node):
        box1 = self.originalBox
        box2 = node.originalBox

        distance_x1 = abs(box1.x1 - box2.x1)
        distance_x2 = abs(box1.x2 - box2.x2)
        distance_y1 = abs(box1.y1 - box2.y1)
        distance_y2 = abs(box1.y2 - box2.y2)
        distance_w = abs(box1.width - box2.width)
        distance_h = abs(box1.height - box2.height)
        distance_x = min(abs(box1.x1 - box2.x2), abs(box1.x2 - box2.x1))
        distance_y = min(abs(box1.y1 - box2.y2), abs(box1.y2 - box2.y1))

        new_x1 = min(box1.x1, box2.x1)
        new_y1 = min(box1.y1, box2.y1)
        new_x2 = max(box1.x2, box2.x2)
        new_y2 = max(box1.y2, box2.y2)
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1

        cost = 0
        cost += max(0, (distance_x - 5) * 0.03)
        cost += min(distance_y, distance_x1 * 0.03)

        return -cost

    def join(self, node):
        return GraphNode(self.originalBox.join(node.originalBox))


class GraphClustering:
    def __init__(self, leaves, clusteringThreshold=-10):
        self.threshold = clusteringThreshold
        self.graphNodes = [GraphNode(l) for l in leaves]
        self.relabelNodes()

        self.graphEdges = [(node1.similarityScore(node2), node1, node2)
                           for node1 in self.graphNodes
                           for node2 in self.graphNodes
                           if node1.id < node2.id]
        self.graphEdges = [e for e in self.graphEdges if e[0] > self.threshold]  # keep only edges with good similarity
        self.resortEdges()

    def relabelNodes(self):
        for i, node in enumerate(self.graphNodes):
            node.id = i

    def resortEdges(self):
        self.graphEdges.sort(key=lambda a: -a[0])

    def joinNodes(self, n1: GraphNode, n2: GraphNode):
        connectedNodes = {}
        notConnectedEdges = []
        for e in self.graphEdges:
            if e[1] == n1 or e[1] == n2:
                connectedNodes[e[2].id] = e[2]
            elif e[2] == n1 or e[2] == n2:
                connectedNodes[e[1].id] = e[1]
            else:
                notConnectedEdges.append(e)

        newNode = n1.join(n2)
        self.graphNodes.append(newNode)

        self.graphEdges = notConnectedEdges
        for v in connectedNodes.values():
            s = newNode.similarityScore(v)
            if s > self.threshold:
                self.graphEdges.append((s, newNode, v))

    def clusterLeaves(self):
        while len(self.graphEdges) > 0:
            _, u, v = self.graphEdges.pop(0)
            self.graphNodes.pop(max(u.id, v.id))
            self.graphNodes.pop(min(u.id, v.id))
            self.joinNodes(u, v)
            self.relabelNodes()
            self.resortEdges()

        return [node.originalBox for node in self.graphNodes]


def main():
    for cv_id in range(1, 40):
        orig = Image.open('../assets/datajobs16.%d.jpg' % cv_id).convert('L')
        copy = orig.convert('RGB')
        draw = ImageDraw.Draw(copy)

        root = SubdivisionNode(np.array(orig))
        subdivideForever(root, 10)

        for box in GraphClustering(getLeaves(root)).clusterLeaves():
            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline="red")

        plt.imshow(copy)
        plt.show()

main()
