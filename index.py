import sys

sys.path.append('/usr/local/lib/python3.6/site-packages')

from segmentLines import segmentLinesFromImage
from line import LineRecognizer

from Classifier import CNNClassifier
# from Classifier import SVMClassifier

from executor.Executor import Executor

from post_processing.Post_Processor import Post_Processor

DEFAULT_LANGUAGE = 'Python'
TEST_IMAGE = 'python_1.png'

lineImages = segmentLinesFromImage(TEST_IMAGE)
recognizer = LineRecognizer(CNNClassifier('./classifiers/bymerge-classifier-10epochs')) # pass in a classifier
processor = Post_Processor(DEFAULT_LANGUAGE)
program = ''

for lineImage in lineImages[::-1]:
    # Returns full predicted line of code
    # Classifier must conform to generic classifier interface.
    lineOfCode = recognizer.recognizeLine(lineImage)
    processedLineOfCode = processor.process_line(lineOfCode)
    program = program + processedLineOfCode + '\n'

# Now run the program
executor = Executor(DEFAULT_LANGUAGE)
output, status = executor.run(program)
print (output)
print (status) # This is what the web service returns. 
