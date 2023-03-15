
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
import glob
warnings.filterwarnings('ignore')

ROOT_DIR="D:\melanoma classification\ISIC_2020_Training_JPEG"

def dataFolder(path,split):
    
    DIR="D:\\melanoma classification";
    if not os.path.exists(DIR+path):
        os.mkdir(DIR+"/"+path)
        for dir in os.listdir(ROOT_DIR):
         os.makedirs(DIR+"/"+path+"/"+dir)

         for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR,dir)),size=(math.floor(split*number_of_images[dir])),replace=False): #70%
            O=os.path.join(ROOT_DIR,dir,img) #path original
            D=os.path.join(DIR+"/"+path+"/"+dir)
            shutil.copy(O,D)
            os.remove(O)

    else:
     print(f"{path}Folder Exists")

number_of_images={}

for dir in os.listdir(ROOT_DIR):
    number_of_images[dir]=len(os.listdir(os.path.join(ROOT_DIR,dir)))

print(number_of_images.items())

dataFolder("train2",0.7)
dataFolder("test2",1)