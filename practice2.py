
import pandas as pd

import shutil 
import os
import numpy as np
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import shutil 
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import transforms
#count the number of images in respective classes (0 & 1)
# df = pd.read_csv("ISIC_2020_Training_GroundTruth.csv")
# img_names = df['image_name']
# labels = df['target']
# train_dict = dict(zip(img_names, labels))

# print(len(train_dict))

# ROOT_DIR="D:\\melanoma classification\\train2\\train"
# number_of_train_images={}

# D1="D:\\melanoma classification\\Melanoma_train2"
# D2="D:\\melanoma classification\\NotMelanoma_train2"

# for dir in os.listdir(ROOT_DIR):
#    d=os.path.join(ROOT_DIR,dir)
#    Str = dir[:len(dir)-4]
#    if(train_dict[Str]):
#         D=os.path.join(D1+"/"+dir)
#         shutil.copy(d,D1)
       
#    else:
#         D=os.path.join(D2+"/"+dir)
#         shutil.copy(d,D2)
img = Image.open('ISIC_0149568.jpg')
img_path=os.path.join('.','ISIC_0149568.jpg')
img = cv2.imread(img_path)
norm_img = np.zeros((800,800))
final_img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
D='D:\\melanoma classification'
img_path2=img_path[:1]
# final_img.save(D)  
E=os.path.join(D+"/"+img_path2)
   #  final_img.save(D)  
cv2.imwrite(E,final_img)
# cv2.imwrite(final_img,D)
        




