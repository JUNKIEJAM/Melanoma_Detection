
import pandas as pd

import shutil 
import os
#count the number of images in respective classes (0 & 1)
df = pd.read_csv("ISIC_2020_Training_GroundTruth.csv")
img_names = df['image_name']
labels = df['target']
train_dict = dict(zip(img_names, labels))

print(len(train_dict))

ROOT_DIR="D:\\melanoma classification\\train2\\train"
number_of_train_images={}

D1="D:\\melanoma classification\\Melanoma_train2"
D2="D:\\melanoma classification\\NotMelanoma_train2"

for dir in os.listdir(ROOT_DIR):
   d=os.path.join(ROOT_DIR,dir)
   Str = dir[:len(dir)-4]
   if(train_dict[Str]):
        D=os.path.join(D1+"/"+dir)
        shutil.copy(d,D1)
       
   else:
        D=os.path.join(D2+"/"+dir)
        shutil.copy(d,D2)
        




