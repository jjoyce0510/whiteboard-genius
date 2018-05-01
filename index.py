import sys
import cv2
sys.path.append('/usr/local/lib/python3.6/site-packages')

from classifiers.classifier import CNNClassifier
from classifiers.classifier import SVMClassifier

from segmenter.line_segmenter import LineSegmenter
from recognizer.line_recognizer import LineRecognizer
from postprocessor.postprocessor import PostProcessor
from executor.Executor import Executor

DEFAULT_LANGUAGE = 'Python'
TEST_IMAGE_NAME = "./images/python_1.png"

lineImages = LineSegmenter().getLinesFromImage(TEST_IMAGE_NAME)
recognizer = LineRecognizer(CNNClassifier('./classifiers/bymerge-allsymbols-3epochs')) # pass in a classifier
processor = PostProcessor(DEFAULT_LANGUAGE)
program = ''

for image in lineImages:
    # Returns full predicted line of code
    cv2.imshow("image", image.getImage())
    cv2.waitKey()
    lineOfCode = recognizer.recognizeLine(image)
    processedLineOfCode = processor.process_line(lineOfCode)
    program = program + processedLineOfCode + '\n'

print(program)
# Now run the program
# executor = Executor(DEFAULT_LANGUAGE)
# output, status = executor.run(program)
# print (output)
# print (status) # This is what the web service returns.
