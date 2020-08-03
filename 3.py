#######################################################
###############                         ###############
###############  @Nimesha Muthunayake   ###############
###############  Reg: 209359G           ###############
###############  Date:08-03-2020        ###############
###############  M.Sc 2020              ###############
###############       (Q3)              ###############
#######################################################
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# Data loading part goes here
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# reshape the train dataset
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# normalize the pixel values from - scaled 255
# convert from integers to floats
X_train = X_train.astype('float32')
X_train /= 255

# convert from integers to floats
X_test = X_test.astype('float32')
X_test /= 255

# convert  to categorical
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

batch_size = 64 #Setting batch size

model = Sequential()

# model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))

model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


#########################################################################################
###                                                                                   ###
###        Adding random Nosie** to the images noise_factor = 0.5                     ###
###                                                                                   ###
#########################################################################################
noise_factor = 0.5
X_train = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test  = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train = np.clip(X_train, 0., 1.)
X_test  = np.clip(X_test, 0., 1.)

#########################################################################################
###                                                                                   ###
###               With data augmentation to prevent overfitting                       ###                         
###                                                                                   ###
#########################################################################################
datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False, 
        zca_whitening=False, 
        rotation_range=10, 
        zoom_range = 0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1,  
        horizontal_flip=False, 
        vertical_flip=False)  

#datagen.flow setings
train_aug = datagen.flow(X_train, Y_train, batch_size=batch_size)
test_aug = datagen.flow(X_test, Y_test, batch_size=batch_size)

epochs = 20 #seetiing number of epochs

#fit and train the model 
history = model.fit_generator(train_aug,
                              epochs = epochs,
                              steps_per_epoch = X_train.shape[0] // batch_size,
                              validation_data = test_aug,
                              validation_steps = X_test.shape[0] // batch_size)