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


model = VGG16(weights='imagenet', include_top=False)
model.summary()
# vgg16_feature_list = []
file_name_and_feature = dict()
folder_path = '/home/harsh/Projects/data/all_face_detect_cropped'

for filename in os.listdir(folder_path):
    feature_list_of_this_image = []
    file_name = folder_path + '/' + filename

    img = image.load_img(os.path.join(folder_path, filename), target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    # vgg16_feature_list.append(vgg16_feature_np.flatten())
    feature_list_of_this_image.append(vgg16_feature_np.flatten())
    file_name_and_feature[file_name] = feature_list_of_this_image

print(list(file_name_and_feature.values()))

p = PCA(random_state=728)
vgg16_pca = p.fit_transform(list(file_name_and_feature.values()))
print(vgg16_pca.shape)


#Dont use DBSCAN, doesnt work in our case
# dbscan = DBSCAN(eps=0.3)
# dbscan.fit_predict(vgg16_pca)
# print(dbscan.labels_)


#Use agglomerative
# dendogram = sch.dendrogram(sch.linkage(vgg16_pca, method = "ward"))
# plt.title("<Title for X-axis>")
# plt.xlabel("<Title for Y-axis>")
# plt.ylabel("Euclidean distance")
# plt.show()
#
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(vgg16_pca)
print(hc.labels_)
# print(file_name_and_feature.keys())
# Visualising the clusters
# plt.scatter(vgg16_pca[y_hc == 0, 0], vgg16_pca[y_hc == 0, 1], s = 100, c = "red", label = "Cluster 1")
# plt.scatter(vgg16_pca[y_hc == 1, 0], vgg16_pca[y_hc == 1, 1], s = 100, c = "blue", label = "Cluster 2")
# plt.scatter(vgg16_pca[y_hc == 2, 0], vgg16_pca[y_hc == 2, 1], s = 100, c = "green", label = "Cluster 3")
# plt.scatter(vgg16_pca[y_hc == 3, 0], vgg16_pca[y_hc == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
# plt.scatter(vgg16_pca[y_hc == 4, 0], vgg16_pca[y_hc == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
# plt.legend()
# plt.show()