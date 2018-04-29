import classifiers.Classifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.models import load_model
import scipy.io as sio
from keras.models import Sequential
from keras.layers.core import Activation, Dropout
from keras.layers import Dense, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# from sklearn.cross_validation import KFold


seed = 7
np.random.seed(seed)

def loadDataset(path):
	# Load the dataset
	mat = sio.loadmat(path)
	mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
	#print(mapping)

	# Extract the features and labels
	features = np.array(mat['dataset'][0][0][0][0][0][0], 'int16')
	labels = np.array(mat['dataset'][0][0][0][0][0][1], 'int')

	features = features.reshape(features.shape[0], 784)
	labels = np_utils.to_categorical(labels, 47)

	return features,labels

def create_model():
	model = Sequential()
	model.add(Dense(512, input_shape=(784,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(47))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model

x, y = loadDataset('../matlab/emnist-bymerge.mat')

kf = KFold(n_splits=10)

cv_scores = []
for train, test in kf.split(x, y):
	model = None
	model = create_model()
	model.fit(x[train], y[train], batch_size=128, epochs=5, verbose=1)
	scores = model.evaluate(x[test], y[test], verbose=1)
	# cvscores.append(scores[1] * 100)
print(scores)
# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
# model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(model, x, y, cv=kfold)
# print(results.mean())

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# for train, test in kfold.split(x, y):
# 	# follows model of classifiers/test

# 	model.fit(x[train], y[train], batch_size=128, epochs=5, verbose=0)

# 	scores = model.evaluate(x[test], y[test], verbose=0)
# 	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# 	cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
