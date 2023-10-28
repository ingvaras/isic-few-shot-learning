import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Dense

labels = pd.read_csv("data/ISIC_2019_Training_GroundTruth.csv")

print(labels.head())

model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'weights/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))
model.add(Dense(9, activation = 'softmax'))

model.layers[0].trainable = False