#######################################################
###############                         ###############
###############  @Nimesha Muthunayake   ###############
###############  Reg: 209359G           ###############
###############  Date:07-29-2020        ###############
###############  M.Sc 2020              ###############
###############       (Q1)              ###############
#######################################################

###  1. Develop a deep learning model for image classification. Include the following in your report. (5 marks)
###  a. Explanation to your model, design decisions, training-test dataset descriptions and what other factors were considered to improve your model.
###  b. Accuracy of the model at the end of each epoch.

# baseline cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# load train and test dataset
def loade_data():
	# load dataset
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	# reshape dataset to have a single channel
	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
	# one hot encode target values
	Y_train = to_categorical(Y_train)
	Y_test = to_categorical(Y_test)
	return X_train, Y_train, X_test, Y_test

# scale pixels
def scale_data(train, test):
	# convert from integers to floats
	train_float = train.astype('float32')
	test_float = test.astype('float32')
	# normalize to range 0-1
	train_float = train_float / 255.0
	test_float = test_float / 255.0
	# return normalized images
	return train_float, test_float

# define cnn model
def define_cnn():
	cnn = Sequential()
	cnn.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	cnn.add(MaxPooling2D((2, 2)))
	cnn.add(Flatten())
	cnn.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	cnn.add(Dense(10, activation='softmax'))
	# compile cnn
	opt = SGD(lr=0.01, momentum=0.9)
	cnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return cnn

# eveluate the cnn model wirh epochs=10 , batch_size is used as 32
def evaluate_cnn(X_train, Y_train, X_test, Y_test):
		scores, summary = list(), list()
		cnn = define_cnn()	
		result = cnn.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
		_, accuracy = cnn.evaluate(X_test, Y_test, verbose=0)
		print('Accuracy for the CNN model = %.3f' % (accuracy * 100.0))
		scores.append(accuracy)
		summary.append(result)
		return scores, summary

# plot diagnostic learning curves
def plot_accuracy(summary):
	for i in range(len(summary)):	
		#pyplot.subplot(2, 1, 2)
		pyplot.title('Model accuracy for each epochs')
		pyplot.plot(summary[i].history['accuracy'], color='red', label='train',linewidth=5)
	pyplot.show()



def MAIN_():

	X_train, Y_train, X_test, Y_test = loade_data()
	X_train, X_test = scale_data(X_train, X_test)
	scores, summary = evaluate_cnn(	X_train, Y_train, X_test, Y_test)
	plot_accuracy(summary)
	


MAIN_()