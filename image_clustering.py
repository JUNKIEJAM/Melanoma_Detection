#Normalization->lab->grayscale->
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

# ROOT_DIR1="D:\\melanoma classification\\train_rgb\\Melanoma"
ROOT_DIR1="D:\\melanoma classification\\train\\Melanoma"
ROOT_DIR2="D:\\melanoma classification\\train_rgb\\Not_Melanoma"
D1="D:\\melanoma classification\\train_image_seg\\Melanoma"
D2="D:\\melanoma classification\\train_image_seg\\Not_Melanoma"

#Normalization
# for dir in os.listdir(ROOT_DIR2): 
#     img_path=os.path.join(ROOT_DIR2,dir)
#     img = cv2.imread(img_path)
#     norm_img = np.zeros((800,800))
#     final_img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
#     D=os.path.join(D2+"/"+dir)
#    #  final_img.save(D)  
#     cv2.imwrite(D,final_img)

# for dir in os.listdir(ROOT_DIR1): 
#     img_path=os.path.join(ROOT_DIR1,dir)
#     img = cv2.imread(img_path)
#     norm_img = np.zeros((800,800))
#     final_img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
#     D=os.path.join(D1+"/"+dir)
#    #  final_img.save(D)     
#     cv2.imwrite(D,final_img)

#GrayScale
# for dir in os.listdir(D2):
#    img_path=os.path.join(D2,dir)
#    img = Image.open(img_path).convert('RGB')
#    srgb_p = ImageCms.createProfile("sRGB")
#    lab_p  = ImageCms.createProfile("LAB")
#    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
#    Lab = ImageCms.applyTransform(img, rgb2lab)
#    L, a, b = Lab.split()
#    D=os.path.join(D2+"/"+dir)
#    os.remove(img_path)
#    L.save(D)     #saved grayscale image

#GrayScale
# for dir in os.listdir(ROOT_DIR1):
#    img_path=os.path.join(ROOT_DIR1,dir)
#    img = Image.open(img_path).convert('RGB')
#    srgb_p = ImageCms.createProfile("sRGB")
#    lab_p  = ImageCms.createProfile("LAB")
#    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
#    Lab = ImageCms.applyTransform(img, rgb2lab)
#    L, a, b = Lab.split() 
#    D=os.path.join(D1+"/"+dir)
#    os.remove(img_path)
#    L.save(D)     #saved grayscale image


# #K-Means Clustering

# # for dir in os.listdir(ROOT_DIR2):
# #    img_path=os.path.join(ROOT_DIR2,dir)
# #    img = Image.open(img_path)
# #    pixel_vals = img.reshape((-1,3))
# #    pixel_vals = np.float32(pixel_vals)
# #    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
# #    k = 2
# #    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# #    centers = np.uint8(centers)
# #    segmented_data = centers[labels.flatten()]
# #    segmented_image = segmented_data.reshape((img.shape))
# #    D=os.path.join(D2+"/"+dir)
# #    os.remove(img_path)
# #    segmented_image.save(D)     #K-Means Done (K=2)


# for dir in os.listdir(ROOT_DIR1):
#        img_path=os.path.join(ROOT_DIR1,dir)
#        img= Image.open(img_path)
#        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#        ret, thresh =     cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#        kernel = np.ones((2,2),np.uint8) 
#        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#        kernel = np.ones((6,6),np.uint8)
#        dilate = cv2.dilate(opening,kernel,iterations=3)
#        blur = cv2.blur(dilate,(15,15))
#        ret, thresh =     cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        contours, hierarchy =     cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#        cnt = max(contours, key=cv2.contourArea)
#        h, w = img.shape[:2]
#        mask = np.zeros((h, w), np.uint8)
#        cv2.drawContours(mask, [cnt],-1, 255, -1)
#        res = cv2.bitwise_and(img, img, mask=mask)
#        D=os.path.join(D1+"/"+dir)
#        os.remove(img_path)
#        img.save(D)     #K-Means Done (K=2)

# for dir in os.listdir(ROOT_DIR2):
#        img_path=os.path.join(ROOT_DIR2,dir)
#        img= Image.open(img_path)
#        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#        ret, thresh =     cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#        kernel = np.ones((2,2),np.uint8) 
#        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#        kernel = np.ones((6,6),np.uint8)
#        dilate = cv2.dilate(opening,kernel,iterations=3)
#        blur = cv2.blur(dilate,(15,15))
#        ret, thresh =     cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        contours, hierarchy =     cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#        cnt = max(contours, key=cv2.contourArea)
#        h, w = img.shape[:2]
#        mask = np.zeros((h, w), np.uint8)
#        cv2.drawContours(mask, [cnt],-1, 255, -1)
#        res = cv2.bitwise_and(img, img, mask=mask)
#        D=os.path.join(D2+"/"+dir)
#        os.remove(img_path)
#        segmented_image.save(D)     

#Morphology
def hair_remove(image):
    # convert image to grayScale
#     grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    
    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    
    return final_image


for dir in os.listdir(D1):
    img_path=os.path.join(D1,dir)
    img= np.array(Image.open(img_path))
    image_resize = cv2.resize(img,(1024,1024))
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(image_resize, cv2.MORPH_BLACKHAT, kernel)
#     plt.subplot(l, 5, (i*5)+3)
#     plt.imshow(blackhat)
#     plt.axis('off')
#     plt.title('blackhat : '+ image_name)
    # intensify the hair countours in preparation for the inpainting 
    ret,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
#     plt.subplot(l, 5, (i*5)+4)
#     plt.imshow(threshold)
#     plt.axis('off')
#     plt.title('threshold : '+ image_name)
    # inpaint the original image depending on the mask
    final_image = cv2.inpaint(image_resize,threshold,1,cv2.INPAINT_TELEA)
#     plt.subplot(l, 5, (i*5)+5)
#     plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.title('final_image : '+ image_name)
    final_image = hair_remove(image_resize)
    D=os.path.join(D1+"/"+dir)
    os.remove(img_path)
#     final_image.save(D)
    cv2.imwrite(D,final_image)