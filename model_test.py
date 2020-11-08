import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras import backend as K
from models import *
from data_prepossing import load_dataset
from keras.models import load_model
from train import *
K.set_image_data_format('channels_last')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
num_classes = 5

img = cv2.imread('test_/113.jpg')
imgRes = cv2.resize(img,(128,128))

X_temp = []
X_temp.append(imgRes)
X = np.asarray(X_temp)
X = X/255


model = InceptionResNetV2(input_shape=imgRes,classes=num_classes,weights=None)
model.load_weights('pretrain_model/inception_resnet_v2.hdf5')

y = model.predict_classes(X)
classno = np.ndarray.tolist(y)

K.clear_session()

dict = {0:'Perfect Road' , 1:'Mild Bad Road' , 2:'Severly Bad Road', 3:'Good Road',4:'Water on Surface'}
objectClass = dict[classno[0]]
print(objectClass)


font = cv2.FONT_HERSHEY_TRIPLEX
tsv=50

# # Thresolding_for_detect_cracks
# #img = cv2.resize(img,(1024,720))
# grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# retval,threshold = cv2.threshold(grayscaled,tsv,255,cv2.THRESH_BINARY)
# #cv2.putText(threshold, 'Thres_value:100',(100,100), font, 2, (210,255,0), 5, cv2.LINE_AA)

# cv2.putText(threshold, objectClass,(100,100), font, 2, (127,250,0), 5, cv2.LINE_AA)
# #cv2.rectangle(threshold,(20,200),(600,400),(255,255,255), 5)



# cv2.imshow('Threshold',threshold)
# print("Threshold value for Road Detecttion:",tsv)

# prediction Road Condition 
#cv2.putText(img, objectClass,(100,900), font, 3, (133, 58, 224), 5, cv2.LINE_AA)
cv2.putText(img, objectClass,(100,500), font, 2, (133, 58, 224), 5, cv2.LINE_AA)
#cv2.rectangle(img,(20,200),(600,400),(255,255,255), 5)
cv2.imshow('Prediction',img)



cv2.waitKey(0)
cv2.destroyAllWindows()