from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import glob
import cv2
from face_detect import facechop


model = VGG16(weights='imagenet', include_top=False)
# vgg16_feature_list = []
file_name_and_feature = dict()

path = "/home/abha/Celeb_faces/train/all"
images = glob.glob(path + "/*")

for imgpath in images[0:]:
    feature_list_of_this_image = []
    imagename = os.path.basename(imgpath)

    cropped = facechop(imgpath)
    if len(cropped) == 1 or len(cropped) == 2:
        cropped_img = cropped[0]
    else:
        continue
    img = cv2.resize(cropped_img, (224,224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)

    flattened = vgg16_feature_np.flatten()
    feature_list_of_this_image.append(flattened)
    file_name_and_feature[imagename] = feature_list_of_this_image

    # fc.append(facechop(imgpath))

# print(len(fc))
print(len(file_name_and_feature))
keys = list(file_name_and_feature.keys())
print(file_name_and_feature[keys[0]][0].shape)
