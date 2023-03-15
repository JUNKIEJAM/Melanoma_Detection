# path='D:\melanoma classification\ISIC_2020_Test_JPEG\ISIC_2020_Test_Input'

import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
import glob
warnings.filterwarnings('ignore')

def dataFolder(path,split):

    if not os.path.exists("./"+path):
        os.mkdir("./"+path)
        for dir in os.listdir(ROOT_DIR):
         os.makedirs("./"+path+"/"+dir)

         for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR,dir)),size=(math.floor(split*number_of_images[dir])-5),replace=False): #70%
            O=os.path.join(ROOT_DIR,dir,img) #path original
            D=os.path.join("./"+path+"/",dir)
            shutil.copy(O,D)
            os.remove(O)

    else:
     print(f"{path}Folder Exists")

#count the number of images in respective classes (0 & 1)

ROOT_DIR="D:\melanoma classification\ISIC_2020_Training_JPEG\train"
number_of_images={}

for dir in os.listdir(ROOT_DIR):
    number_of_images[dir]=len(os.listdir(os.path.join(ROOT_DIR,dir)))

print(number_of_images.items())

dataFolder("train",0.7)
dataFolder("test",1)

#we will split data into 70:15:15 (train:validate:test)

#Model Building
import keras
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAvgPool2D

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator




#CNN Model

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))

model.add(Conv2D(filters=36,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics=['accuracy'])


#Prepring our data using Data Generator


def preprocessingImages(path):
    #input: Path
    #Output: Pre processed images

    image_data=ImageDataGenerator(zoom_range=0.2,shear_range=0.2,rescale=1/255,horizontal_flip=True)

    image=image_data.flow_from_directory(directory=path,target_size=(224,224),batch_size=32,class_mode='binary')

    return image

path="C:/Users/PRITHESH DWIVEDI/OneDrive/Documents/GitHub/Melanoma/train"
train_data=preprocessingImages(path)






