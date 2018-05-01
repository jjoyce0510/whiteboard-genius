import sys
from sys import argv

sys.path.append('/usr/local/lib/python3.6/site-packages')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.layers.core import Activation, Dropout
from keras.layers import Dense, Flatten
from keras.utils import np_utils
import scipy.io as sio
import numpy as np
from keras.optimizers import Adam as Adam
from keras.layers.advanced_activations import LeakyReLU

import os
import cv2

import h5py

# need to extern this shit
mapping = {0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65, 
			11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74, 20: 75, 
			21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84, 30: 85, 
			31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101, 
			40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116
		}

#need final 1 for channel dim
INPUT_SHAPE = (28, 28, 1)
num_classes = 126 #58 # emnist + custom + math_symbols
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
	# mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}

	# Extract the features and labels
	features = np.array(mat['dataset'][0][0][0][0][0][0], 'float32')
	labels = np.array(mat['dataset'][0][0][0][0][0][1], 'float32')

	#threshold the training images
	features_th = []
	for feature in features:
		ret, feature_th = cv2.threshold(feature, 170, 255, cv2.THRESH_BINARY) #adjust this. 
		# feature_th = cv2.adaptiveThreshold(feature, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		features_th.append(feature_th)

	features = np.array(features_th)

	#ensure correct shape
	features = features.reshape(features.shape[0], IMG_X, IMG_Y, 1)

	return features, labels

def loadCustomData(path):
	labels = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
	img_files = []
	for l in labels:
		img_files = img_files + [(f, l) for f in os.listdir(os.path.join(path, l)) if f[:2] == 't_']

	x = []
	y = []

	label_dict = {
		'and' : ord('&'),
		'colon' : ord(':'),
		'dash' : ord('-'),
		'doub_quote' : ord('"'),
		'percent' : ord('%'),
		'period' : ord('.'),
		'semi' : ord(';'),
		'single_quote' : ord("'"),
		'times' : ord('*'),
		'xor' : ord('^')
	}

	# update the mapping dictionary
	mapping.update(label_dict)

	for img, label in img_files:
		p = os.path.join(path, label)
		p = os.path.join(p, img)
		image = cv2.imread(p)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		image_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		image_crop = cv2.resize(image_thresh, (28, 28), interpolation=cv2.INTER_AREA)
		image = np.array(image_crop, 'float32')
		image = image.reshape(28, 28, 1)

		x.append(image)
		y.append(mapping[label]) # label needs to line up with the mapping dict

	x = np.array(x)
	y = np.array(y)

	y = y.reshape(y.shape[0], 1)

	return x, y

def loadMathData(path):
	labels = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
	img_files = []
	for l in labels:
		img_files = img_files + [(f, l) for f in os.listdir(os.path.join(path, l))]

	label_dict = {
		'-' : ord('-'),
		',' : ord(','),
		'!' : ord('!'),
		'(' : ord('('),
		')' : ord(')'),
		']' : ord(']'),
		'[' : ord('['),
		'{' : ord('{'),
		'}' : ord('}'),
		'+' : ord('+'),
		'=' : ord('=')
	}

	# update the mapping dictionary
	mapping.update(label_dict)

	x = []
	y = []

	for img, label in img_files:
		p = os.path.join(path, label)
		p = os.path.join(p, img)
		image = cv2.imread(p)
		# cv2.imshow('hi', image)
		# cv2.waitKey()

		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		image_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		image_crop = cv2.resize(image_thresh, (28, 28), interpolation=cv2.INTER_AREA)
		image = np.array(image_crop, 'float32')
		image = image.reshape(28, 28, 1)

		x.append(image)
		y.append(mapping[label]) # label needs to line up with the mapping dict

	x = np.array(x)
	y = np.array(y)

	y = y.reshape(y.shape[0], 1)

	return x, y

def flip_rotate(features):
	temp_array = []
	for feature in features:
		temp_feature = np.rot90(np.fliplr(feature))
		temp_array.append(temp_feature)

	return np.array(temp_array, 'uint8')

def extractHogFeatures(features):
	# Extract the hog features
	list_hog_fd = []
	for feature in features:
		fd = hog(np.rot90(np.fliplr(feature.reshape((28, 28)))), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		list_hog_fd.append(fd)
		d = np.array(list_hog_fd, 'float64')
		print(fd.shape)

	hog_features = np.array(list_hog_fd, 'float64')

	return hog_features

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

def create_model():
	# calculate mean and standard deviation
	mean_px = x_train.mean().astype(np.float32)
	std_px = x_train.std().astype(np.float32)
	# function to normalize input data
	def norm_input(x): return (x-mean_px)/std_px

	#use sequential model
	model = Sequential([
        Lambda(norm_input, input_shape=INPUT_SHAPE, output_shape=INPUT_SHAPE),
        Conv2D(32, (3,3)),
        LeakyReLU(),
        BatchNormalization(axis=1),
        Conv2D(32, (3,3)),
        LeakyReLU(),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Conv2D(64, (3,3)),
        LeakyReLU(),
        BatchNormalization(axis=1),
        Conv2D(64, (3,3)),
        LeakyReLU(),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

	model.summary()

	model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

	return model

def train_model(model, x_train, y_train, x_test, y_test):
	model.fit(x_train, y_train, batch_size=batch_size, epochs=2, verbose=1, validation_data=(x_test, y_test))

if __name__ == '__main__':

	classifier_name = 'bymerge-allsymbols-3epochs'

	x_sym, y_sym = loadCustomData('../symbol_training/')

	x_math, y_math = loadMathData('../math_symbols/extracted_images/')
	# load dataset
	features, labels = loadDataset('../matlab/emnist-bymerge')

	# flip + rotate the images
	features = flip_rotate(features)

	# extract hog features
	# features = extractHogFeatures(features)

	features = np.concatenate([features, x_sym])
	features = np.concatenate([features, x_math])

	print(labels.shape, y_sym.shape)
	labels = np.concatenate([labels, y_sym])
	print(labels.shape, y_math.shape)
	labels = np.concatenate([labels, y_math])
	print(labels.shape)

	# split into
	(x_train, y_train), (x_test, y_test) = split_data(features,labels)

	# label0 = y_train[20]
	# feature0 = x_train[20]

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	y_train = y_train.astype('uint8')
	y_test = y_test.astype('uint8')
	x_train /= 255
	x_test /= 255

	y_train = y_train.T[0]
	# print(int(y_train.max()))
	# print(len(mapping))
	# print(mapping)
	# print(y_train)

	y_train = np_utils.to_categorical(y_train, num_classes) # we get the wrong shape out of here!!!

	y_test = np_utils.to_categorical(y_test, num_classes)
	#create classifier
	model = create_model()

	train_model(model, x_train, y_train, x_test, y_test)

	acc = model.evaluate(x_test, y_test, verbose=0)[0]

	# write_info(classifier_name, x_train[0], acc)

	# Save the classifier
	model.save("../classifiers/" + classifier_name)
