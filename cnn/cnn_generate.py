import sys
from sys import argv

sys.path.append('/usr/local/lib/python3.6/site-packages')

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout
from keras.layers import Dense, Flatten
from keras.utils import np_utils
# from keras.datasets import mnist
# from sklearn.externals import joblib
# import pickle
import scipy.io as sio
import numpy as np
# from skimage.feature import hog

import cv2

import h5py


mapping = {0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74, 20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84, 30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101, 40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116}

#need final 1 for channel dim
INPUT_SHAPE = (28, 28, 1)
num_classes = 47
EPOCHS = 5
batch_size = 128
IMG_X = 28
IMG_Y = 28

def eval():
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

def loadDataset(path):
	# Load the dataset
	mat = sio.loadmat(path)
	mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
	#print(mapping)

	# Extract the features and labels
	features = np.array(mat['dataset'][0][0][0][0][0][0], 'int16')
	labels = np.array(mat['dataset'][0][0][0][0][0][1], 'int')

	#ensure correct shape do I need?
	features = features.reshape(features.shape[0], IMG_X, IMG_Y,1)

	return features,labels

def flip_rotate(features):
	temp_array = []
	for feature in features:
		temp_feature = np.rot90(np.fliplr(feature))
		temp_array.append(temp_feature)

	return np.array(temp_array, 'uint8')

# def extractHogFeatures(features):
# 	# Extract the hog features
# 	list_hog_fd = []
# 	for feature in features:
# 		fd = hog(np.rot90(np.fliplr(feature.reshape((28, 28)))), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
# 		list_hog_fd.append(fd)
# 		d = np.array(list_hog_fd, 'float64')
# 		print(fd.shape)

# 	hog_features = np.array(list_hog_fd, 'float64')

# 	return hog_features

#splits data into seperate test and training arrays
def split_data(features,labels):
	x_train_list = []
	x_test_list = []
	y_train_list = []
	y_test_list = []

	count = 0
	for feature in features:
		if count % 10 == 0:
			x_test_list.append(feature)
			y_test_list.append(labels[count])
		else: 
			x_train_list.append(feature)
			y_train_list.append(labels[count])
		count = count + 1


	x_test = np.array(x_test_list, 'uint8')
	x_train = np.array(x_train_list, 'uint8')
	y_test = np.array(y_test_list, 'uint8')
	y_train = np.array(y_train_list, 'uint8')

	return (x_train, y_train), (x_test, y_test)

def train_model(x_train, y_train, x_test, y_test):
	#use sequential model
	model = Sequential()
	#add 2d convolutional layer
	# model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=INPUT_SHAPE))
	# #add 2d max pooling layer
	# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	# #add 2d convolutional layer
	# model.add(Conv2D(64, (5, 5), activation='relu'))
	# #add 2d max pooling layer
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	# #flatten output
	# model.add(Flatten())
	# #add full connected layer
	# model.add(Dense(1000, activation='relu'))
	# #add output layer
	# model.add(Dense(num_classes, activation='softmax'))
	# #select loss func and optomizer
	# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])


	model = Sequential()
	model.add(Dense(512, input_shape=(784,)))
	model.add(Activation('relu'))
	                           
	model.add(Dropout(0.2))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	            

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS, verbose=1, validation_data=(x_test, y_test))

	return model

if __name__ == '__main__':
	#get command line arguements
	data_path = argv[1]
	classifier_name = argv[2]

	#set up command line parser
	#parser = argparse.ArgumentParser(description='Process some integers.')
	#parser.add_argument('-t', type=str, nargs='+', help='Training Data')
	#parser.add_argument('--foo', help='foo of the %(prog)s program')
	


	#load dataset
	features, labels = loadDataset(data_path)

	#flip + rotate the images
	features = flip_rotate(features)
	
	#extract hog features
	# features = extractHogFeatures(features)

	#split into 
	(x_train, y_train), (x_test, y_test) = split_data(features,labels)


	print('x_train shape:', x_train.shape)
	# print(x_train.shape, 'train samples')
	print(x_test.shape, 'test samples')


	label0 = y_train[20]
	feature0 = x_train[20]
	#print(feature0)
	#print(feature0.shape)
	#print(label0)
	#cv2.imshow("image", feature0);
	#cv2.waitKey(0)

	#convert the data to the right type
	x_train = x_train.reshape(x_train.shape[0], 784)
	x_test = x_test.reshape(x_test.shape[0], 784)

	print(x_train.shape)
	print(x_test.shape)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)

	model = train_model(x_train,y_train,x_test,y_test)

	# Save the classifier
	model.save("../classifiers/" + classifier_name)
