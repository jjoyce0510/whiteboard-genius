import cv2
import random
#maybe use sliding window; issue with the other is lack of spaces.
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
        text = ""
        for word in wordImages:
            cv2.imshow("word", word.getImage())
            cv2.waitKey()
            graph = self.CharGraph(word, self.classifier)
            text = text + graph.getCharacters() + " "

        return text

    def splitLineByWhitespace(self, lineImage):
        # 1. Threshold
        ret, im_th = cv2.threshold(lineImage, 160, 255, cv2.THRESH_BINARY) #adjust this.
        # cv2.imshow("thresholded", im_th)

        # 2. Expand
        dilation = cv2.erode(im_th,(11, 11), iterations = 7)
        # cv2.imshow("dilated", dilation)

        # 3. Blur
        im_blurred = cv2.GaussianBlur(dilation, (9, 89), 0)
        # cv2.imshow("blurred", im_blurred)

        # 4. Threshold
        ret, im_th = cv2.threshold(im_blurred, 240, 255, cv2.THRESH_BINARY) #adjust this.
        # cv2.imshow("re-thresholded", im_th)

        bordersize=10
        im_th_border=cv2.copyMakeBorder(im_th, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=255 )
        orig_img_border=cv2.copyMakeBorder(lineImage, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # Find contours in image
        _, ctrs, hier = cv2.findContours(im_th_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        words = []

        # For each rectangular region, calculate HOG features and predict
        for rect in rects:
            if rect[0] is 0 and rect[1] is 0:
                continue
                im_gray_border[adjustedY:adjustedY+rect[3], rect[0]:rect[0]+rect[2]]
            words.append(LineImage(orig_img_border[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]))
            #cv2.rectangle(im_th_border, (rect[0], rect[1]), (rect[0]+rect[2], rect[1] + rect[3]), (63, 191, 118), 2)
            #cv2.rectangle(orig_img_border, (rect[0], rect[1]), (rect[0]+rect[2], rect[1] + rect[3]), (63, 191, 118), 2)

        # cv2.imshow("Resulting Image with Rectangular ROIs", im_th_border)
        # cv2.imshow("YEP", orig_img_border)
        # cv2.waitKey()

        return words

    class CharGraph:
        SAMPLING_WIDTH_IN_PX = 5
        adjacencyMap = {}
        chars = None
        line = None
        classifier = None

        def __init__(self, line, classifier):
            self.line = line
            self.classifier = classifier
            self.build()

        # PUBLIC
        def getCharacters(self):
            # See if we've already computed
            if self.chars is not None:
                return self.chars
            else:
                path = self.getAverageLongestPath()
                # print (path)
                return self.getCharactersFromPath(path, str(self.line.getWidth()))

        # PRIVATE
        def build(self):
            # Basically go through the image, every 10 px create a node in our graph
            # Algorithm based on https://cse.sc.edu/~songwang/document/wacv13c.pdf
            currentX = 0
            lineWidth = self.line.getWidth()

            # For each SAMPLING_WIDTH_IN_PX increment, create edges between current position until end on line w/ SAMPLING_WIDTH_IN_PX increment
            while currentX < lineWidth:
                self.createEdgesBetweenCoordinates(currentX, lineWidth, self.SAMPLING_WIDTH_IN_PX)
                currentX = currentX + self.SAMPLING_WIDTH_IN_PX

        def createEdgesBetweenCoordinates(self, firstCoordinate, finalCoordinate, samplingWidth):
            firstNodeLabel = self.coordinateToLabel(firstCoordinate)
            if firstNodeLabel not in self.adjacencyMap:
                self.adjacencyMap[firstNodeLabel] = [] # empty edge array

            connectedCoordinate = firstCoordinate + samplingWidth

            while connectedCoordinate <= finalCoordinate:
                # Add the edge between first node and connected node
                if connectedCoordinate - firstCoordinate <= 5:
                    connectedCoordinate = connectedCoordinate + samplingWidth
                    continue

                self.adjacencyMap[firstNodeLabel].append(self.createEdge(firstCoordinate, connectedCoordinate))
                # Increment to the next node.
                connectedCoordinate = connectedCoordinate + samplingWidth

            if connectedCoordinate > finalCoordinate:
                connectedCoordinate = finalCoordinate
                self.adjacencyMap[firstNodeLabel].append(self.createEdge(firstCoordinate, connectedCoordinate))

        def createEdge(self, origin, dest):
            newEdgeImage = self.line.getImage()[:, origin:dest]
            prediction, newEdgeWeight = self.classifier.classify(newEdgeImage)
            # cv2.imshow(str(prediction) + ' : ' + str(newEdgeWeight), newEdgeImage)
            # cv2.waitKey()
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
            # print(finalNode)
            # Basically you want to trace back from the final node to build a string
            currNode = path[finalNode]
            reversedString = self.getCharactersFromPathRecursive(path, currNode, '')
            return reversedString[::-1]

        def getCharactersFromPathRecursive(self, path, currNode, currString):
            if currNode[1] is None:
                # End of list
                return currString

            # print(currNode[0])
            # print edges
            # print "edge: " + currNode[1].getOrigin() + " " + currNode[1].getDestination() + " " + str(currNode[1].getWeight())
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
