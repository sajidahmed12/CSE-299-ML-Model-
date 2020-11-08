import random
import numpy as np
import cv2
import pandas
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import glob
from tqdm import tqdm


def data_extractor(path):	
	readpath ='dataset/watered_road/*.jpg'
	labels = 'labels/wlabels.csv'
	objectClass = 'watered_road'

	images = glob.glob(readpath)
	labelfile = open(labels,'w')

	for image in images:
	    labelfile.write(image+','+objectClass+'\n')

	labelfile.close()



def shuffle_fn(path):
	labels = 'labels/labels.csv'
	shuffled_labels = 'labels/shuffled_labels.csv'

	labelfile = open(labels, "r")
	lines = labelfile.readlines()
	labelfile.close()
	random.shuffle(lines)

	shufflefile = open(shuffled_labels, "w")
	shufflefile.writelines(lines)
	shufflefile.close()


def create_npz(path):

	labels = 'labels/shuffled_labels.csv'
	npzfile = 'labels/labels.npz'

	df = pandas.read_csv(labels)

	rows = df.iterrows()

	X_temp = []
	Y_temp = []

	for row in tqdm(rows):
	    image = row[1][0]
	    img = cv2.imread(image)
	    img = cv2.resize(img,(128,128))
	    imageClass = row[1][1]
	    X_temp.append(img)
	    Y_temp.append(imageClass)


	encoder = LabelEncoder()
	encoder.fit(Y_temp)
	encoded_Y = encoder.transform(Y_temp)
	Y = np_utils.to_categorical(encoded_Y)

	np.savez(npzfile, X_train=X_temp,Y_train=Y)


def load_dataset():
	npzfile = 'labels/labels.npz'
	dataset =  np.load(npzfile)
	x_train = dataset['X_train']
	y_train = dataset['Y_train']
	x = x_train/255

	return x, y_train