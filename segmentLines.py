# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
from line import LineImage
import numpy as np

# TODO make as general as possible
# TODO add deskewing
# TODO add line breaking support

# Segments a single rectangle into multiple, may end up running on all lines, probably should use histogram
def segmentLinesFromRectangle(rect, img):
    return None

# Can tweak process here if necessary
# Assumption is that valid lines will be detected the majority of the time (bad for one liners)
def processRects(rects, img):
    MINIMUM_ASPECT_RATIO = 1
    MINIMUM_INSTANCES_FOR_IQR = 5
    MIN_SPLIT_DEV = 3

    widths = []
    heights = []
    filtered_rects = []

    for rect in rects:
        aspect_ratio = rect[2]/rect[3]

        # Filter by aspect ratio, don't include extraneous in mean
        if aspect_ratio > MINIMUM_ASPECT_RATIO:
            print rect
            widths.append(rect[2])
            heights.append(rect[3])
            filtered_rects.append(rect)

    avg_width = np.mean(np.array(widths))
    avg_height = np.mean(np.array(heights))

    print(avg_width)
    print(avg_height)

    new_avg_height = avg_height
    # Remove outliers using IQR if enough samples
    if len(filtered_rects) > MINIMUM_INSTANCES_FOR_IQR:
        np_heights = np.array(heights)
        np_heights = np.sort(np_heights)
        q75, q25 = np.percentile(np_heights, [75, 25])
        iqr = q75 - q25
        outlier_factor = 1.5 * iqr
        # Get rid of statistical outliers in terms of height (width doesn't tell us much)
        heights_without_outliers = [h for h in heights if abs(avg_height - h) < outlier_factor]
        new_avg_height = np.mean(np.array(heights_without_outliers))

    print ("Avg height w/o outliers: " + str(new_avg_height))

    std_width = np.std(np.array(widths))
    std_height = np.std(np.array(heights_without_outliers))

    print("St dev width: " + str(std_width))
    print("St dev height: " + str(std_height))

    output_rects = []

    # Process remaining rectangles
    for rect in filtered_rects:
        # Remove samples more than 1.5 stdevs from height mean, since line heights should be similar
        rect_height = rect[3]

        if (rect_height > new_avg_height + MIN_SPLIT_DEV * std_height):
            #output_rects.append(rect)
            # Try to break the rects
            seg_rects = segmentLinesFromRectangle(rect, img)
            if seg_rects and len(seg_rects) > 1:
                # More than 1 segmented rect, add to filtered rects for normal processing
                for seg in seg_rects:
                    filtered_rects.append(seg)
                    continue
            else:
                print ("Segmentation Error: Unable to parse lines from rect.")

        if (rect_height > new_avg_height - 1.5 * std_height) and (rect_height < new_avg_height + 3.5 * std_height):
            output_rects.append(rect)

    return output_rects

# Returns array of type Line
def segmentLinesFromImage(imageName):
    ### Image pre-processing/segmentation pipeline
    RECT_HEIGHT_ADJUSTMENT_FACTOR = 1.25
    MIN_IMAGE_RESIZE_WIDTH = 1500 # px
    PIXEL_THRESHOLD_1 = 150
    PIXEL_THRESHOLD_2 = 240
    PIXEL_THRESHOLD_3 = 240

    # Read the input image
    im = cv2.imread(imageName)
    height, width, _ = im.shape

    if width > MIN_IMAGE_RESIZE_WIDTH:
        im = cv2.resize(im, (0,0), fx=0.25, fy=0.25)

    # 1. Grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("grayscale initial", im_gray)

    # 2. Binarize
    ret, im_b = cv2.threshold(im_gray, PIXEL_THRESHOLD_1, 255, cv2.THRESH_BINARY) #adjust this.

    # 3. Blur
    im_blurred = cv2.GaussianBlur(im_b, (89, 5), 0)
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
    #cv2.imshow("dilated", dilation)

    # 8. Erosion
    erosion = cv2.erode(dilation,(11, 11),iterations = 5)
    #cv2.imshow("erosion", erosion)

    # 9. Add whitespace border
    bordersize=20
    im_border=cv2.copyMakeBorder(erosion, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=255 )
    im_original_border=cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=(255, 255, 255) )
    #cv2.imshow("bordered", im_border)
    im_gray_border=cv2.copyMakeBorder(im_gray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=255 )

    # Find contours in image
    _, ctrs, hier = cv2.findContours(im_border.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    rects = processRects(rects, im_border)

    lines = []

    # For each rectangular region, calculate HOG features and predict
    for rect in rects:
        print (rect)
        # Draw the rectangles
        adjustedHeight = int(rect[3] * RECT_HEIGHT_ADJUSTMENT_FACTOR) # 1.5 * height of rect
        adjustedY = int(rect[1] - 0.25 * rect[3])
        if (adjustedY < 0):
            adjustedY = 0


        # add Line to lines
        lines.append(LineImage(im_gray_border[adjustedY:adjustedY+rect[3], rect[0]:rect[0]+rect[2]]))

        cv2.rectangle(im_border, (rect[0], adjustedY), (rect[0]+rect[2], adjustedY + adjustedHeight), (63, 191, 118), 2)
        #cv2.rectangle(im_original_border, (rect[0], adjustedY), (rect[0]+rect[2], adjustedY + adjustedHeight), (63, 191, 118), 2)

    #cv2.imshow("Resulting Image with Rectangular ROIs", im_border)
    #cv2.imshow("Output blobs with Rectangles", im_original_border)

    #return lines to caller
    return lines
