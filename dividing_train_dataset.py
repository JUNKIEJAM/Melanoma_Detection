import pandas as pd
import shutil 
import os

df = pd.read_csv("ISIC_2020_Test_Metadata.csv")
img_names = df['image_name']
labels = df['target']
train_dict = dict(zip(img_names, labels))

print(len(train_dict))

ROOT_DIR="D:\\melanoma classification\\ISIC_2020_Training_JPEG\\train"
number_of_train_images={}

D1="D:\\melanoma classification\\Melanoma_train"
D2="D:\\melanoma classification\\Not_Melanoma_train"
for dir in os.listdir(ROOT_DIR):
    # print(dir)
   d=os.path.join(ROOT_DIR,dir)
   Str = dir[:len(dir)-4]
   if(train_dict[Str]):
        # helper(d,"Melanoma")
        # O=os.path.join(ROOT_DIR,dir,img) #path original
        D=os.path.join(D1+"/"+dir)
        shutil.copy(d,D1)
        # os.remove(O)

   else:
        D=os.path.join(D2+"/"+dir)
        shutil.copy(d,D2)