from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

model = VGG16(weights='imagenet', include_top=False)
model.summary()
# vgg16_feature_list = []
file_name_and_feature = dict()
folder_path = '/home/abha/Celeb_faces/train/all_face_detect_cropped'

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


# print(vgg16_feature_list[0].shape)
test_filepath = '/home/abha/Celeb_faces/train/all_face_detect_cropped/httpftqncomymusicLxZeltonjohnjpg_30.jpg'
print(file_name_and_feature[test_filepath][0].shape)