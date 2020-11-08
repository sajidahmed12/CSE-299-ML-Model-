from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D

def CNN_model(img_dim, num_classes):

	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode= 'valid' , input_shape=img_dim,activation= 'relu'))
	model.add(Convolution2D(32, (3, 3),activation= 'relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Convolution2D(64,(3, 3),activation= 'relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(512, activation= 'relu' ))
	model.add(Dropout(0.45))
	model.add(Dense(num_classes, activation= 'softmax' ))
	model.compile(loss= 'categorical_crossentropy' , optimizer= 'Adamax' , metrics=[ 'accuracy' ])
	
	return model