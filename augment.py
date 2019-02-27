import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import numpy as np
from sklearn.cluster import DBSCAN
from keras.layers.core import Flatten
from keras.models import Sequential

model_VGG16 = VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))

train_dir = '/home/harsh/Projects/data/train'
image_size = 224
train_batchsize = 10

def create_datagen():
    train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')

    return train_generator


def covnet_transform(covnet_model, raw_images):
    # Pass our training data through the network
    pred = covnet_model.predict(raw_images)

    # Flatten the array
    flat = pred.reshape(raw_images.shape[0], -1)

    return flat

# Reference for the below function:
# Part 6.2 of https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751


def extract_features(sample_count):  # sample_count is the no. of training images
    features = np.zeros(shape=(sample_count, 7 * 7 * 512))  # Must be equal to the output of the convolutional base
    # labels = np.zeros(shape=(sample_count))

    generator = create_datagen()
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = covnet_transform(model_VGG16, inputs_batch)
        # print(labels_batch)
        features[i * train_batchsize: (i + 1) * train_batchsize] = features_batch
        # labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * train_batchsize >= sample_count:
            break
    return features

# print(extract_features(92).shape)
db = DBSCAN(eps=0.3).fit_predict(extract_features(103))
print(db)