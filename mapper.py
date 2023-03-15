import pandas as pd

df = pd.read_csv("ISIC_2020_Training_GroundTruth.csv")
img_names = df['image_name']
labels = df['target']
train_dict = dict(zip(img_names, labels))

# print(train_dict)