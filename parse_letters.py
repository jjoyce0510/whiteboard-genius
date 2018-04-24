# Import the modules
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

IMAGE_NAME = 'scanned_test.png'

CURR_FOLDER = 'symbol_training/doub_quote'
files = [f for f in listdir(CURR_FOLDER) if isfile(join(CURR_FOLDER, f))]
images = [f for f in files if '.png' in f and 'j' in f]

currNum = 328
# Returns array of type Line
def segmentLinesFromImage(imageName):

    print imageName
    global currNum

    ### Image pre-processing/segmentation pipeline
    RECT_HEIGHT_ADJUSTMENT_FACTOR = 1.25
    PIXEL_THRESHOLD_1 = 220
    PIXEL_THRESHOLD_2 = 240
    PIXEL_THRESHOLD_3 = 240

    # Read the input image
    im = cv2.imread(imageName)
    height, width, _ = im.shape

    # 1. Grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("grayscale initial", im_gray)

    # 2. Binarize
    ret, im_b = cv2.threshold(im_gray, PIXEL_THRESHOLD_1, 255, cv2.THRESH_BINARY) #adjust this.

    # 3. Blur
    im_blurred = cv2.GaussianBlur(im_b, (59, 121), 0)
    #cv2.imshow("blurred", im_blurred)

    # 4. Threshold
    ret, im_th = cv2.threshold(im_blurred, PIXEL_THRESHOLD_2, 255, cv2.THRESH_BINARY) #adjust this.
    #cv2.imshow("re-thresholded", im_th)

    # 5. Blur
    im_reblurred = cv2.GaussianBlur(im_th, (51, 1), 0)
    #cv2.imshow("re-blurred", im_reblurred)

    # 6. Threshold
    ret, im_reth = cv2.threshold(im_reblurred, PIXEL_THRESHOLD_3, 255, cv2.THRESH_BINARY) #adjust this.
    #cv2.imshow("rere-thresholded", im_reth)

    # 7. Dilate
    dilation = cv2.dilate(im_reth,(11, 11),iterations = 6)
    cv2.imshow("dilated", dilation)

    # 8. Erosion
    erosion = cv2.erode(dilation,(11, 11),iterations = 5)
    cv2.imshow("erosion", erosion)

    cv2.waitKey()

    # 9. Add whitespace border
    bordersize=20
    im_border=cv2.copyMakeBorder(erosion, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=255 )
    im_original_border=cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=(255, 255, 255) )
    im_gray_border=cv2.copyMakeBorder(im_gray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=255 )

    # Find contours in image
    _, ctrs, hier = cv2.findContours(im_border.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    lines = []

    # For each rectangular region, calculate HOG features and predict
    for rect in rects:
        # Draw the rectangles
        adjustedHeight = int(rect[3] * RECT_HEIGHT_ADJUSTMENT_FACTOR) # 1.5 * height of rect
        adjustedY = int(rect[1] - 0.25 * rect[3])
        if (adjustedY < 0):
            adjustedY = 0

        lines.append(im_original_border[adjustedY:adjustedY+adjustedHeight, rect[0]:rect[0]+rect[2]])

        cv2.rectangle(im_border, (rect[0], adjustedY), (rect[0]+rect[2], adjustedY + adjustedHeight), (63, 191, 118), 2)
        #cv2.rectangle(im_original_border, (rect[0], adjustedY), (rect[0]+rect[2], adjustedY + adjustedHeight), (63, 191, 118), 2)


    for image in lines:
        cv2.imshow("resulting.", image)
        key = cv2.waitKey()
        if key is 115:
            # save image
            fileName = CURR_FOLDER + '/' + "t_" + str(currNum) + '.png'
            cv2.imwrite(fileName, image)
            currNum = currNum + 1


for image in images:
    segmentLinesFromImage(CURR_FOLDER + '/' + image)
