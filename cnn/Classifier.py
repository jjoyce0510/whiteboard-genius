#Classifier.py
#Contains classifier class
import cv2
from keras.models import load_model
from keras.layers import Activation, Dense
# from skimage.feature import hog
import numpy as np

# edit this mapping
mapping = {0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74, 20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84, 30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101, 40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116}


# Loads image
def loadImage(path):
    return cv2.imread(path)

# Loads the classifier
def loadClassifer(path):
    return load_model(path)

def preprocessImage(image):
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    #Threshold the image
    ret, im_th = cv2.threshold(im_gray, 170, 255, cv2.THRESH_BINARY_INV) #adjust this.

    return im_th

class CNNClassifier(object):
    def __init__(self, model_path):
        self.model = loadClassifer(model_path)
        #self.model.add(Dense(47, activation='softmax'))

    def classify(self, image):
        # We need to reshape to be 28x28
        image = cv2.resize(image, (28, 28))
        cv2.imshow("hello", image)
        cv2.waitKey()
        #do we need this?
        image_array = np.array([image], 'uint8')

        #do we need this?
        #Preprocess image
        #processedImage = preprocessImage(image)

        #do we need this?
        #roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

        #reshape to have 1 for channel dim
        image_array = image_array.reshape(image_array.shape[0],28,28,1)

        probs = self.model.predict(image_array)
        predictionIndex = np.argmax(probs)
        predictionProbability = probs[0][predictionIndex]
        prediction = chr(mapping[predictionIndex])

        print (predictionProbability)
        print prediction

        return prediction, predictionProbability
