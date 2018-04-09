import sys

sys.path.append('/usr/local/lib/python3.6/site-packages')

from segmentLines import segmentLinesFromImage
from line import LineRecognizer
from cnn.Classifier import CNNClassifier

DEFAULT_LANGUAGE = 'C++'
TEST_IMAGE = "exampleCode.jpg"

lineImages = segmentLinesFromImage(TEST_IMAGE)
program = ''
recognizer = LineRecognizer(CNNClassifier('./classifiers/cnn-bymerge-E5.h5')) # pass in a classifier

for lineImage in lineImages:
    # Returns full predicted line of code
    # Classifier must conform to generic classifier interface.
    lineOfCode = recognizer.recognizeLine(lineImage)
    print(lineOfCode)
    program = program + lineOfCode

print(program)

# runProgram(program, DEFAULT_LANGUAGE)
