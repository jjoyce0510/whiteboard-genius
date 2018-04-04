#Classifier.py
#Contains classifier class

import cv2
from keras.models import load_model
from skimage.feature import hog
import numpy as np

mapping = {0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74, 20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84, 30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101, 40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116}

# Loads image
def loadImage(path):
    return cv2.imread(input_image_path)
    
# Loads the classifier
def loadClassifer(path):
    return load_model(path)

class Classifier(object):
    def __init__(self, model_path):
        self.model = loadClassifer(model_path)
    
    def classify(image):
        #do we need this?
        image_array = np.array([image], 'uint8')

        #do we need this?
        #Preprocess image
        #processedImage = preprocessImage(image)

        #do we need this?
        #roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

        #reshape to have 1 for channel dim
        roi_array = roi_array.reshape(roi_array.shape[0],28,28,1)

        probs = self.model.predict(roi_array)
        prediction = chr(mapping[np.argmax(probs)])
    
        return prediction, probs
