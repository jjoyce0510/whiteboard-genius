# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import scipy.io as sio
from collections import Counter

# Load the dataset
mat = sio.loadmat("emnist-bymerge")
mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
# print mapping

# Extract the features and labels
features = np.array(mat['dataset'][0][0][0][0][0][0], 'int16')
labels = np.array(mat['dataset'][0][0][0][0][0][1], 'int')

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(np.rot90(np.fliplr(feature.reshape((28, 28)))), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')


# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, labels.ravel())

# Save the classifier
joblib.dump(clf, "emnist_cls.pkl", compress=3)
