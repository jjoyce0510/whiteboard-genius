import sys

sys.path.append('/usr/local/lib/python3.6/site-packages')

from segmentLines import segmentLinesFromImage
from line import LineRecognizer

from Classifier import CNNClassifier
from Classifier import SVMClassifier

DEFAULT_LANGUAGE = 'Python'
TEST_IMAGE = "python_1.png"

lineImages = segmentLinesFromImage(TEST_IMAGE)
program = ''
recognizer = LineRecognizer(CNNClassifier('classifiers/bymerge-classifier-5epochs')) # pass in a classifier

for lineImage in lineImages:
    # Returns full predicted line of code
    # Classifier must conform to generic classifier interface.
    lineOfCode = recognizer.recognizeLine(lineImage)
    print(lineOfCode)
    program = program + lineOfCode

print(program)

# runProgram(program, DEFAULT_LANGUAGE)
