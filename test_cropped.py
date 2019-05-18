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
from face_detect import getCroppedImages
from keras.models import model_from_json
from inception_resnet_v1 import *
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


model_path = '/home/abha/Facenet/facenet_model.json'
# model = model_from_json(open(model_path, "r").read())
model = InceptionResNetV1()
#Download below weights
#pre-trained weights https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
model.load_weights('/home/abha/Facenet/facenet_keras_weights.h5')



path = "/home/abha/Celeb_faces/train/all"
file_path_and_cropped_image = getCroppedImages(path)
file_path_and_feature = dict()

for file_path in file_path_and_cropped_image.keys():
    cropped_img = file_path_and_cropped_image[file_path]
    img = cv2.resize(cropped_img, (160,160))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    facenet_feature = model.predict(img_data)
    print("done")
    facenet_feature_np = np.array(facenet_feature)
    facenet_feature_np_flattened = facenet_feature_np.flatten()
    file_path_and_feature[file_path] = list(facenet_feature_np_flattened)

    # fc.append(facechop(imgpath))

# print(len(fc))
print(len(file_path_and_feature))
keys = list(file_path_and_feature.keys())
print(len(file_path_and_feature[keys[0]]))
