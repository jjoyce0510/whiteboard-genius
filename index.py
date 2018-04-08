from segmentLines import segmentLinesFromImage
from line import LineRecognizer

DEFAULT_LANGUAGE = 'C++'
TEST_IMAGE = "exampleCode.jpg"

lineImages = segmentLinesFromImage(TEST_IMAGE)
program = ''
for lineImage in lineImages:
    #Returns full predicted line of code
    recognizer = LineRecognizer(None) # pass in a classifier
    lineOfCode = recognizer.recognizeLine(lineImage)
    print lineOfCode
    program = program + lineOfCode

print program

# runProgram(program, DEFAULT_LANGUAGE)
