import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

data = pd.read_csv('train.csv')

x = data.iloc[:,1:]
y = data.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state = 1212)
# print(x_train.shape)     #(33600, 784)
# print(x_test.shape)      #(8400, 784)

num_pixels=784

x_train = x_train.as_matrix().reshape(33600, 784)
x_test = x_test.as_matrix().reshape(8400, 784)

# Feature Normalization 
x_train = x_train.astype('float32');
x_test= x_test.astype('float32');

x_train /= 255;
x_test /= 255;

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# print(y_test.shape[1])   #10
num_classes = y_test.shape[1]

# define baseline model
#simple neural network with 1 hidden layer
#a rectifier activation function is used for the neurons in the hidden layer
#a softmax activation function is used on the output layer to turn the outputs into probability-like values and allow one class of the 10
#to be selected as the model's output prediction.
#logarithmic function is used as the loss function(called categical_crossentropy in Keras)
#ADAM gradient descent algorihm is used to learn the weights.
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))