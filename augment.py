import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import numpy as np

model = VGG16(include_top=False)

train_dir = '/home/abha/Celeb_faces/train'
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

# Reference for the below function:
# Part 6.2 of https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751


def extract_features(sample_count):  # sample_count is the no. of training images
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    # labels = np.zeros(shape=(sample_count))

    generator = create_datagen()
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch)
        features[i * train_batchsize: (i + 1) * train_batchsize] = features_batch
        # labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * train_batchsize >= sample_count:
            break
    return features

print(extract_features(93).shape)