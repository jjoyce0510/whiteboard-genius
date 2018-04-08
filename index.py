from segmentLines import segmentLinesFromImage
from line import LineRecognizer
from cnn.Classifier import CNNClassifier

DEFAULT_LANGUAGE = 'C++'
TEST_IMAGE = "exampleCode.jpg"

lineImages = segmentLinesFromImage(TEST_IMAGE)
program = ''
for lineImage in lineImages:
    # Returns full predicted line of code
    # Classifier must conform to generic classifier interface.
    recognizer = LineRecognizer(CNNClassifier('./classifiers/cnn-bymerge-E5.h5')) # pass in a classifier
    lineOfCode = recognizer.recognizeLine(lineImage)
    print lineOfCode
    program = program + lineOfCode

print program

# runProgram(program, DEFAULT_LANGUAGE)
