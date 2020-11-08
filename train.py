from models import ResNeXt,CNN_model
from utils import *
import numpy as np
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.applications import *

#import all model class here
def load_model(num_classes):

	img_dim = (128,128,3)
	#model =  CNN_model(img_dim, num_classes)
	#model =  ResNet50(input_shape=img_dim,classes=num_classes,weights=None)
	#model =  InceptionResNetV2(input_shape=img_dim,classes=num_classes,weights=None)
	#model =  MobileNetV2(input_shape=img_dim,classes=num_classes,weights=None)
	#model =  Xception(input_shape=img_dim,classes=num_classes,weights=None)
	#model =  ResNet101(input_shape=img_dim,classes=num_classes,weights=None)
	#model =  ResNet152(input_shape=img_dim,classes=num_classes,weights=None)
	#model =  ResNeXt(input_shape=img_dim,classes=num_classes)
	model =  InceptionV3(input_shape=img_dim,classes=num_classes,weights=None)
	#model =  DenseNet121(input_shape=img_dim,classes=num_classes,weights=None)
	#model =  NASNetLarge(input_shape=img_dim,classes=num_classes,weights=None)
	#model =  InceptionV4(input_shape=img_dim,classes=num_classes,weights=None)




	optmize1 = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)
	optmize2 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(loss= 'categorical_crossentropy' , optimizer= optmize1 , metrics=[ 'accuracy' ])
	return model


def preview_model(model):

  	model.summary()


def train_model(x, y, model, seed = 7, epoch=20, batch_size=12):

	np.random.seed(seed)
	K.set_image_data_format('channels_last')

	check  = ModelCheckpoint('pretrain_model/'+model.name+'.hdf5', monitor = 'val_categorical_accuracy' )
	checkpoints = [check]
	print("Training is started using...." +model.name+ " Model")
	history=model.fit(x, y, validation_split = 0.2, nb_epoch=epoch, batch_size=batch_size,verbose=2, callbacks = checkpoints)

	#prediction values
	scores = model.evaluate(x, y)
	np.savetxt('result/'+model.name+'.txt' ,scores,delimiter=',')
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


	# list all data in history
	print(history.history.keys())

	# summarize history for accuracy
	plot_history(history,'acc','val_acc','model_accuracy','epoch','accuracy')

	# summarize history for loss
	plot_history(history,'loss','val_loss','model_loss','epoch','loss')