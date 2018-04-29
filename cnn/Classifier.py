#Classifier.py
#Contains classifier class
import cv2
from keras.models import load_model
from keras.layers import Activation, Dense
# from skimage.feature import hog
import numpy as np

# edit this mapping
mapping = {0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74, 20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84, 30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101, 40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116}
char_to_index = dict((c, i) for i, c in mapping.items())


class CNNClassifier(object):
    def __init__(self, model_path):
        self.model = self.loadClassifer(model_path)

    # Loads image
    def loadImage(self, path):
        return cv2.imread(path)

    # Loads the classifier
    def loadClassifer(self, path):
        return load_model(path)

    def classify(self, image):
        # image = self.loadImage(image)
        image_array = self.pre_process_image(image)
        cv2.imshow('pre processed', image_array[0])
        cv2.waitKey()

        probs = self.model.predict(image_array, verbose = 0)[0]
        prediction = chr(mapping[np.argmax(probs)])
        max_index = char_to_index[ord(prediction)]
        max_probability = probs[max_index]
    
        print(prediction, max_probability)
        return prediction, max_probability

    def pad_image(self, image):
        width = image.shape[0]
        return image

    def pre_process_image(self, image):
        # TODO: check for image size and act accordingly 
        # we may not want to stretch/squeeze too much!
        print(image.shape)
        if image.shape[0] < 7:
            image = pad_image(image)

        # We need to reshape to be 28x28
        image = cv2.resize(image, (28, 28))
        # print(image)
        # Convert to grayscale and apply Gaussian filtering
        # im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(image, (5, 5), 0)

        #Threshold the image
        ret, thresh_image = cv2.threshold(im_gray, 140, 255, cv2.THRESH_BINARY_INV) #adjust this.

        image_cropped = cv2.resize(thresh_image, (28, 28), interpolation=cv2.INTER_AREA)

        image_array = np.array([thresh_image], 'float32')

        #do we need this?
        #roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

        image_array /= 255
        
        # reshape to have 1 for channel dim
        image_array = image_array.reshape(image_array.shape[0], 28, 28, 1)

        return image_array


if __name__ == '__main__':
    cnn = CNNClassifier('../classifiers/bymerge-classifier-15epochs')
    image = '../test.jpg'
    print(cnn.classify(image))

