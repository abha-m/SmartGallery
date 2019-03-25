from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
import os
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from keras.models import model_from_json


# model = VGG16(weights='imagenet', include_top=False)
# model.summary()

#Download below json
#facenet model structure: https://github.com/serengil/tensorflow-101/blob/master/model/facenet_model.json
model_path = '/home/harsh/keras/facenet_model.json'
model = model_from_json(open(model_path, "r").read())

#Download below weights
#pre-trained weights https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
model.load_weights('/home/harsh/keras/facenet_keras_weights.h5')
# model.summary()


# vgg16_feature_list = []
file_name_and_feature = dict()
folder_path = '/home/harsh/Projects/data/all_face_detect_cropped'

feature_list = []
file_path_list = []
for filename in os.listdir(folder_path):
    feature_list_of_this_image = []
    file_name = folder_path + '/' + filename

    img = image.load_img(os.path.join(folder_path, filename), target_size=(160, 160))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    feature_list.append(vgg16_feature_np.flatten())
    file_path_list.append(os.path.join(folder_path, filename))
    # vgg16_feature_list.append(vgg16_feature_np.flatten())
    # feature_list_of_this_image.append(list(vgg16_feature_np.flatten()))
    # print(feature_list_of_this_image)
    # file_name_and_feature[file_name] = feature_list_of_this_image

# print(feature_list[0])


# p = PCA(random_state=728)
# vgg16_pca = p.fit_transform(feature_list)
# print(vgg16_pca.shape)


# Dont use DBSCAN, doesnt work in our case
# dbscan = DBSCAN(eps=0.3)
# dbscan.fit_predict(feature_list)
# print(dbscan.labels_)

#Use agglomerative
# dendogram = sch.dendrogram(sch.linkage(feature_list, method = "ward"))
# plt.title("<Title for X-axis>")
# plt.xlabel("<Title for Y-axis>")
# plt.ylabel("Euclidean distance")
# plt.show()

# hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
# y_hc = hc.fit_predict(feature_list)
# predictions = hc.labels_

# Segregate into cluster
# For now hardcoded 5 clusters
# clusters = [[],[],[],[],[]]
# no_of_clusters = predictions.max()
# for i in range(0,len(predictions)):
#     clusters[predictions[i]].append(i)
# print(clusters)
# print(hc.labels_)
# print(file_path_list)

# Printing clusters and their file path
# for cluster in clusters:
#     for i in cluster:
#         print(file_path_list[i])
#     print("-x-x-x-x-x-x-x-x-")


# Visualising the clusters
# plt.scatter(feature_list[y_hc == 0, 0], feature_list[y_hc == 0, 1], s = 100, c = "red", label = "Cluster 1")
# plt.scatter(feature_list[y_hc == 1, 0], feature_list[y_hc == 1, 1], s = 100, c = "blue", label = "Cluster 2")
# plt.scatter(feature_list[y_hc == 2, 0], feature_list[y_hc == 2, 1], s = 100, c = "green", label = "Cluster 3")
# plt.scatter(feature_list[y_hc == 3, 0], feature_list[y_hc == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
# plt.scatter(feature_list[y_hc == 4, 0], feature_list[y_hc == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
# plt.legend()
# plt.show()