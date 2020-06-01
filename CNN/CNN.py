import numpy as np
import keras
from keras.datasets import mnist
from keras import backend as be
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

batch_size = 128
num_classes = 10
epochs = 12

#input dimensions
rows = 28
cols = 28

#loading the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#data pre-processing (tf backend: channels last)
x_train = x_train.reshape(x_train.shape[0] , rows, cols, 1)
x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)
ip_shape = (rows, cols, 1)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("x_train.shape :", x_train.shape)
print("# of training samples :", x_train.shape[0])
print("# of test samples :", x_test.shape[0])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#network architecture
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = ip_shape))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose = 0)

print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
