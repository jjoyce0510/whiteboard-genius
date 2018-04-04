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
        graph = self.CharGraph(line, self.classifier)
        return graph.getCharacters()

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
            #cv2.imshow("edge from " + str(origin) + " to " + str(dest), newEdgeImage)
            #cv2.waitKey()
            #newEdgeWeight, prediction = self.classifier.classifyChar(newEdgeImage) <-- make sure HOG features extracted
            newEdgeWeight = random.uniform(0, 1)
            prediction = 'E'
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
                newAvg = (currAvg + edge.getWeight())/(numVisited + 1)

                if destNode not in pathDict:
                    pathDict[destNode] = [0.0, None] # Avg thus far, edge getting there

                if newAvg > pathDict[destNode][0]:
                    # Better avg, replace
                    pathDict[destNode][0] = newAvg
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