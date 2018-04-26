import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

class LineImage:
    def __init__(self, image):
        self.image = image
        height, width = image.shape
        self.width = width
        self.height = height

    def getImage(self):
        return self.image

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

class LineRecognizer:
    def __init__(self, classifier):
        # we just want a single Line obj, from which we create the CharGraph
        self.classifier = classifier

    def recognizeLine(self, line):
        # First, need to break away unnecessary whitespace/break into single words.
        wordImages = self.splitLineByWhitespace(line.getImage())
        text = ''
        for word in wordImages:
            text = text + self.PixelHistogram(word, self.classifier).getCharacters() + ' '

            #graph = self.CharGraph(word, self.classifier)
            #graph.build()
            #text = text + graph.getCharacters() + " "

        return text

    def splitLineByWhitespace(self, lineImage):
        # 1. Threshold
        ret, im_th = cv2.threshold(lineImage, 160, 255, cv2.THRESH_BINARY) #adjust this.
        #cv2.imshow("thresholded", im_th)

        # 2. Expand
        dilation = cv2.erode(im_th,(11, 11), iterations = 7)
        #cv2.imshow("dilated", dilation)

        # 3. Blur
        im_blurred = cv2.GaussianBlur(dilation, (9, 89), 0)
        #cv2.imshow("blurred", im_blurred)

        # 4. Threshold
        ret, im_th = cv2.threshold(im_blurred, 240, 255, cv2.THRESH_BINARY) #adjust this.
        #cv2.imshow("re-thresholded", im_th)

        bordersize=10
        im_th_border=cv2.copyMakeBorder(im_th, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=255 )
        orig_img_border=cv2.copyMakeBorder(lineImage, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # Find contours in image
        _, ctrs, hier = cv2.findContours(im_th_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        rects = self.sortRectsHorizontally(rects)
        words = []
        print rects
        # For each rectangular region, calculate HOG features and predict
        for rect in rects:
            if rect[0] is 0 and rect[1] is 0:
                continue
                im_gray_border[adjustedY:adjustedY+rect[3], rect[0]:rect[0]+rect[2]]
            words.append(LineImage(orig_img_border[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]))
            #cv2.rectangle(im_th_border, (rect[0], rect[1]), (rect[0]+rect[2], rect[1] + rect[3]), (63, 191, 118), 2)
            #cv2.rectangle(orig_img_border, (rect[0], rect[1]), (rect[0]+rect[2], rect[1] + rect[3]), (63, 191, 118), 2)

        #cv2.imshow("Resulting Image with Rectangular ROIs", im_th_border)
        #cv2.imshow("YEP", orig_img_border)
        #cv2.waitKey()

        return words

    def sortRectsHorizontally(self, rects):
        rects.sort(key=lambda x: x[0])
        return rects


    class PixelHistogram:
        USE_GRAPH = False

        def __init__(self, word, classifier):
            self.word = word
            self.wordImage = self.preprocessWord(word)
            self.classifier = classifier
            self.build()

        def preprocessWord(self, word):
            # Threshold
            ret, thresh_image = cv2.threshold(word.getImage(), 180, 255, cv2.THRESH_BINARY) #adjust this.
            cv2.imshow("thresholded word", thresh_image)

            # Blur
            im_blurred = cv2.GaussianBlur(thresh_image, (1, 3), 0)
            #cv2.imshow("b1", im_blurred)

            # threshold
            ret, thresh_image_2 = cv2.threshold(im_blurred, 220, 255, cv2.THRESH_BINARY) #adjust this.
            #cv2.imshow("t3", thresh_image_2)

            cv2.waitKey()
            return thresh_image

        def build(self):
            # build a pixel-density histogram
            image_array = np.array(self.wordImage, 'int16')
            histogram = np.zeros(len(image_array[0]))

            for row in image_array:
                for index, col in enumerate(row):
                    if not col:
                        histogram[index] = histogram[index] + 1

            print histogram

            # Trim edge whitespace
            zero_splits, histogram = self.getZeroTrimSplits(histogram)
            # Now get split other split candidates
            stats_splits = self.getStatsSplits(histogram) or []
            # Get dead space n_splits first
            dead_splits = self.getPixelThresholdSplits(histogram)
            # adjust
            inner_splits = list(map(lambda x: x + (zero_splits[0]+1 or 0), stats_splits + dead_splits))

            self.splits = zero_splits + inner_splits
            self.splits.sort()
            print(self.splits)

            # Now split the histogram to get the character splits.
            #plt.bar(range(histogram.size), height=histogram)
            #plt.title("Histogram with 'auto' bins")
            #plt.show()

        def getZeroTrimSplits(self, histogram):
            if not len(histogram): return []
            # Trim from head
            leadingZerosCount = 0
            currIndex = 0
            while not histogram[currIndex]:
                leadingZerosCount = leadingZerosCount + 1
                currIndex = currIndex + 1

            trailingZerosCount = 0
            currIndex = len(histogram) - 1
            while not histogram[currIndex]:
                trailingZerosCount = trailingZerosCount + 1
                currIndex = currIndex - 1

            headSplitIndex = leadingZerosCount
            tailSplitIndex = len(histogram) - trailingZerosCount

            return [headSplitIndex - 1, tailSplitIndex], histogram[headSplitIndex:tailSplitIndex]

        def getPixelThresholdSplits(self, histogram):
            return self.splitWordByPixelDensity(histogram, 0)

        def getStatsSplits(self, histogram):
            avg_pixel_density = np.mean(histogram)
            mean_pixel_density = np.median(histogram)
            std_pixel_density = np.std(histogram)

            return []

        def splitWordByPixelDensity(self, histogram, pixelValue):
            # Finds all 0 pixel splits
            candidates = []
            foundCandidate = False
            consecutiveCount = 0
            startingIndex = None
            for index, value in enumerate(histogram):
                if value <= pixelValue:
                    if foundCandidate:
                        consecutiveCount = consecutiveCount + 1
                    else:
                        startingIndex = index
                        foundCandidate = True
                        consecutiveCount = 1
                else:
                    if foundCandidate:
                        middleIndex = int((startingIndex + startingIndex + consecutiveCount) / 2)
                        candidates.append(middleIndex)

                    foundCandidate = False
                    consecutiveCount = 0
                    startingIndex = None
            return candidates

        def getCharacters(self):
            # uses splits to classify all text, returns string of text
            EDGE_BUFFER_PIXELS = 5

            # Use CharGraph longest path to find correct characters
            if self.USE_GRAPH:
                graph = CharGraph(self.wordImage, self.classifier)
                graph.buildFromEdgeList(self.splits) # pass candidate splits
                return graph.getCharacters()

            # Otherwise, just use our default splits
            if len(self.splits):
                characters = ''
                # Predict each character in our splits (NAIVE), edge splits should reflect whitespace.
                for index, value in enumerate(self.splits):
                    if index is not (len(self.splits) - 1):
                        # Get the image reflecting the splits
                        currImage = self.wordImage[:, value:self.splits[index+1]]
                        cv2.imshow("image without border", currImage)

                        borderedImage = cv2.copyMakeBorder(
                            currImage,
                            top=0, bottom=0,
                            left=EDGE_BUFFER_PIXELS,
                            right=EDGE_BUFFER_PIXELS,
                            borderType= cv2.BORDER_CONSTANT,
                            value=255)

                        prediction, probability = self.classifier.classify(borderedImage)
                        characters = characters + prediction
                        cv2.imshow("image with border", borderedImage)
                        cv2.waitKey()

                return characters
            else:
                # Likely a single character, touching both edges of the frame, just classify + move on.
                prediction, probability = self.classifier.classify(self.wordImage)
                return prediction


    class CharGraph:
        SAMPLING_WIDTH_IN_PX = 5
        adjacencyMap = {}
        chars = None
        line = None
        classifier = None

        def __init__(self, line, classifier):
            self.line = line
            self.classifier = classifier

        # PUBLIC
        def getCharacters(self):
            # See if we've already computed
            if self.chars is not None:
                return self.chars
            else:
                path = self.getAverageLongestPath()
                print (path)
                return self.getCharactersFromPath(path, str(self.line.getWidth()))

        # PRIVATE
        def build(self):
            # Basically go through the image, every 10 px create a node in our graph
            # Algorithm based on https://cse.sc.edu/~songwang/document/wacv13c.pdf
            currentX = 0
            lineWidth = self.line.getWidth()

            # For each 10px increment, create edges between current position until end on line w/ 10px increment
            while currentX < lineWidth:
                self.createEdgesBetweenCoordinates(currentX, lineWidth, self.SAMPLING_WIDTH_IN_PX)
                currentX = currentX + self.SAMPLING_WIDTH_IN_PX

        def buildFromEdgeList(self, nodes):
            for index, value in enumerate(nodes):
                if index is not (len(nodes) - 1):
                    # Get the image reflecting the splits
                    currImage = self.line.getImage()[:, value:nodes[index+1]]
                    borderedImage = cv2.copyMakeBorder(
                        self.wordImage,
                        top=0, bottom=0,
                        left=EDGE_BUFFER_PIXELS,
                        right=EDGE_BUFFER_PIXELS,
                        borderType= cv2.BORDER_CONSTANT,
                        value=255)

                    prediction, probability = self.classifier.classify(borderedImage)
                    characters = characters + prediction
                    cv2.imshow("image with border", borderedImage)
                    cv2.waitKey()

        def createEdgesBetweenCoordinates(self, firstCoordinate, finalCoordinate, samplingWidth):
            firstNodeLabel = self.coordinateToLabel(firstCoordinate)
            if firstNodeLabel not in self.adjacencyMap:
                self.adjacencyMap[firstNodeLabel] = [] # empty edge array

            connectedCoordinate = firstCoordinate + samplingWidth

            while connectedCoordinate <= finalCoordinate:
                # Add the edge between first node and connected node
                self.adjacencyMap[firstNodeLabel].append(self.createEdge(firstCoordinate, connectedCoordinate))
                # Increment to the next node.
                connectedCoordinate = connectedCoordinate + samplingWidth

            if connectedCoordinate > finalCoordinate:
                connectedCoordinate = finalCoordinate
                self.adjacencyMap[firstNodeLabel].append(self.createEdge(firstCoordinate, connectedCoordinate))

        def createEdge(self, origin, dest):
            newEdgeImage = self.line.getImage()[:, origin:dest]
            prediction, newEdgeWeight = self.classifier.classify(newEdgeImage)
            #newEdgeWeight = random.uniform(0, 1)
            #prediction = 'E'
            return self.Edge(self.coordinateToLabel(origin), self.coordinateToLabel(dest), newEdgeWeight, prediction)

        def coordinateToLabel(self, coordinate):
            return str(coordinate)

        def getAverageLongestPath(self):
            # Calculate avg longest path here
            # 1. keep track of max avg arriving at each node (and where it came from )
            # 2. at final node track back to the beginning
            # Start at node with label 0
            # TODO: SEGMENT OUT INDIVIDUAL WORDS
            # Somehow keep track of the prediction as well.

            initialNode = str(0) # Label for initial node
            avgLongestPathDict = {}
            avgLongestPathDict[str(0)] = [0.0, None] # avg, prediction
            nodesVisited = 0

            self.getAverageLongestPathRecursive(initialNode, avgLongestPathDict, nodesVisited)

            #print (avgLongestPathDict) # Should be able to trace back from here.

            return avgLongestPathDict

        def getAverageLongestPathRecursive(self, currNode, pathDict, numVisited):

            if currNode not in self.adjacencyMap:
                return

            # Traverse starting at currNode
            edges = self.adjacencyMap[currNode]
            currAvg = pathDict[currNode][0]
            # Traverse all edges
            for edge in edges:
                # edges already have weights, calculate new avg.
                destNode = edge.getDestination()
                # TODO: THIS IS NOT THE CORRECT WEIGHTED AVG.
                newWeightedAvg = currAvg * numVisited
                newWeightedAvg = (newWeightedAvg + edge.getWeight())/(numVisited + 1)

                if destNode not in pathDict:
                    pathDict[destNode] = [0.0, None] # Avg thus far, edge getting there

                if newWeightedAvg > pathDict[destNode][0]:
                    # Better avg, replace
                    pathDict[destNode][0] = newWeightedAvg
                    pathDict[destNode][1] = edge
                    # Only traverse into node if new avg better than old average. (One way traversal is key here.)
                    self.getAverageLongestPathRecursive(destNode, pathDict, numVisited + 1)

        def getCharactersFromPath(self, path, finalNode):
            print(finalNode)
            # Basically you want to trace back from the final node to build a string
            currNode = path[finalNode]
            reversedString = self.getCharactersFromPathRecursive(path, currNode, '')
            return reversedString[::-1]

        def getCharactersFromPathRecursive(self, path, currNode, currString):
            if currNode[1] is None:
                # End of list
                return currString

            print(currNode[0])
            # print edges
            print "edge: " + currNode[1].getOrigin() + " " + currNode[1].getDestination() + " " + str(currNode[1].getWeight())
            currChar = currNode[1].getPrediction()
            currString = currString + currChar
            nextNode = currNode[1].getOrigin()

            return self.getCharactersFromPathRecursive(path, path[nextNode], currString)

        class Edge:
            weight = None
            origin = None
            destination = None
            prediction = None

            def __init__(self, origin, destination, weight, prediction):
                self.origin = origin
                self.destination = destination
                self.weight = weight
                self.prediction = prediction

            def getWeight(self):
                return self.weight

            def getDestination(self):
                return self.destination

            def getOrigin(self):
                return self.origin

            def getPrediction(self):
                return self.prediction
