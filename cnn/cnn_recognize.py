import sys
from sys import argv
sys.path.append('/usr/local/lib/python3.6/site-packages')

import cv2
from keras.models import load_model
from skimage.feature import hog
import numpy as np

mapping = {0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74, 20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84, 30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101, 40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116}


def loadImage(path):
    return cv2.imread(input_image_path)

def loadClassifer(path):
    # Load the classifier
    return load_model(path)

def preprocessImage(image):
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    #cv2.imshow("gray scaled and blurred!", im_gray)

    #Threshold the image
    ret, im_th = cv2.threshold(im_gray, 170, 255, cv2.THRESH_BINARY_INV) #adjust this. 

    #cv2.imshow("thresholded scaled", im_th)

    return im_th


def cropImage(rects,image):
     #For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    rect = rects[0]
    # Draw the rectangles
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    cv2.imshow("thresholded scaled", image)

    leng = int(rect[3] * 1.6)
    ptx = int(rect[1] + rect[3] // 2 - leng // 2)
    pty = int(rect[0] + rect[2] // 2 - leng // 2)

    #could sub leng twice!
    flag = False
    if ptx < 0:
        leng = leng - ptx
        ptx = 0
    if pty < 0:
        leng = leng - pty
        pty = 0


    roi = processedImage[ptx:ptx+leng, pty:pty+leng]

    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    return roi

def classify(roi,model):
    # Calculate the HOG features
    # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    roi_array = np.array([roi], 'uint8')
    
    final = roi_array.reshape(28,28)
    print(final)
    print(final.shape)
    cv2.imshow("final", final)


    roi_array = roi_array.reshape(roi_array.shape[0],28,28,1)
    probs = model.predict(roi_array)
    prediction = chr(mapping[np.argmax(probs)])
    
    return prediction, probs


def locateChar(image):
    # Find contours in the image
    _, ctrs, hier = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    return rects
    

if __name__ == '__main__':
    #get command line arguements
    classifier_path = argv[1]
    input_image_path = argv[2]

    # Read the input image
    image = loadImage(input_image_path)

    # Read the classifier
    model = loadClassifer(classifier_path)

    #Preprocess image
    processedImage = preprocessImage(image)

    #put rectangle around char
    rects = locateChar(processedImage)

    image_cropped = cropImage(rects,image)

    #classify the char image
    prediction, probabilities = classify(image_cropped,model)

    #print all probs
    print(probabilities)
    count = 0
    #for prob in probabilities:
     #   print(chr(mapping[count]), prob)
      #  count = count + 1

    #print max prob
    print(prediction)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    