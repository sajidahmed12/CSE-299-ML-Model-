# -*- coding: utf-8 -*-
"""test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16WTcc3pbiMPR0PzJiNb-kYADMWV3D330
"""



from google.colab import drive
drive.mount('/content/drive')

cd drive/My\ Drive/299_Journal_model/

ls



"""# Test"""

from models import ResNeXt,CNN_model
from utils import *
import numpy as np
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.applications import *

from glob import glob

pretrain_list = []

for i in glob('pretrain_model/*'):
  print(i)
  pretrain_list.append(i)

print(len(pretrain_list))

temp_model = pretrain_list[6]
print(temp_model)

img_dim = (128,128,3)
num_classes = 5

#model =  DenseNet121(input_shape=img_dim,classes=num_classes,weights=None)
#model =  InceptionResNetV2(input_shape=img_dim,classes=num_classes,weights=None)
#model =  InceptionV3(input_shape=img_dim,classes=num_classes,weights=None)
#model =  MobileNetV2(input_shape=img_dim,classes=num_classes,weights=None)
#model =  NASNetLarge(input_shape=img_dim,classes=num_classes,weights=None)
#model =  ResNet50(input_shape=img_dim,classes=num_classes,weights=None)
model =  CNN_model(img_dim, num_classes)
#model =  Xception(input_shape=img_dim,classes=num_classes,weights=None)

#model =  ResNet101(input_shape=img_dim,classes=num_classes,weights=None)
#model =  ResNet152(input_shape=img_dim,classes=num_classes,weights=None)
#model =  ResNeXt(input_shape=img_dim,classes=num_classes)
#model =  InceptionV4(input_shape=img_dim,classes=num_classes,weights=None)

optmize1 = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)
model.compile(loss= 'categorical_crossentropy' , optimizer= optmize1 , metrics=[ 'accuracy' ])

"""### weight load"""

# model.load_weights('pretrain_model/sequential_1.hdf5')
model.load_weights(temp_model)

"""### model test"""

from glob import glob

test_img = []

for i in glob('dataset/testing/*'):
  #print(i)
  test_img.append(i)

print(test_img)

import cv2

def load_img(img_path):
  temp = cv2.imread(img_path)
  imgRes = cv2.resize(img,(128,128))

  temp = np.asarray([imgRes])
  temp = temp/255

  return temp

from tqdm import tqdm

dic = {  
          '0':'Perfect Road' , 
          '1':'Mild Bad Road' , 
          '2':'Severly Bad Road', 
          '3':'Good Road',
          '4':'Water on Surface'
       }

class_list = ['Perfect Road','Mild Bad Road','Severly Bad Road','Good Road','Water on Surface']

# for i in range(len(test_img)):
  # X = load_img(test_img[i])

y_pred = []

for i in tqdm(test_img):

  X = load_img(i)
  y = model.predict(X)
  classno = np.argmax(y,axis=1)

  #K.clear_session()
  #print(classno[0])
  y_pred.append(classno[0])
  # objectClass = class_list[classno[0]]
  # print(objectClass)

from sklearn.metrics import f1_score

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

# y_true = [0, 0, 0]
# y_pred = [0, 0, 0]

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
f1_none = f1_score(y_true, y_pred, average=None)

print("F1 Macro : ",f1_macro)
print("F1 Micro : ",f1_micro)
print("F1 Weighted : ",f1_weighted)
print("F1 None : ",f1_none)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

y_true = [[13,1,1,0,2],
     [3,9,6,0,1],
     [0,0,16,2,0],
     [0,0,0,13,0],
     [0,0,0,0,15]]

class_label = ['Perfect Road','Mild Bad Road','Severly Bad Road','Good Road','Water on Surface']
df_cm = pd.DataFrame(y_true, class_label,class_label)
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)  #for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 9})# font size

plt.show()

'''
0 - bad
1 - normal
2 - good
'''
my_class = ['bad','normal','good']
my_confusion = np.zeros((3,3), dtype=int)

y_true = [0, 1, 1, 2, 0, 2]
y_pred = [0, 2, 1, 1, 0, 2]

'''
    0 1 2
  0 2 0 0
  1 0 1 1
  2 0 1 1
'''
print(my_confusion)

for i,j in zip(y_true,y_pred):
    # print(i,j)
    my_confusion[i][j]+=1
print(my_confusion)

df_cm = pd.DataFrame(my_confusion,my_class,my_class)
df_cm

