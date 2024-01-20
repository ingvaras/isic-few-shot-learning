import os

import numpy as np
import tensorflow as tf
from sklearn.neighbors import KernelDensity
from utils import accuracy, f1, balanced_accuracy

IMAGE_SIZE = 128
SHOTS = 2
CLASSES = ['SCC', 'AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

def load_image(file_path):
    img = tf.keras.utils.load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    img_array = tf.keras.utils.img_to_array(img).flatten()
    return img_array

kde_models = {}
for category in CLASSES:
    kde_models[category] = KernelDensity(bandwidth=0.2, kernel='gaussian')
    directory_path = os.path.join('data/train', category)
    i = 0
    images = []
    for filename in os.listdir(directory_path):
        i += 1
        if i > SHOTS:
            kde_models[category].fit(images)
            break
        file_path = os.path.join(directory_path, filename)
        images.append(load_image(os.path.join(directory_path, filename)))

def predict(sample):
    return np.argmax([kde_models[cat].score_samples([sample]) for cat in CLASSES])


true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0
positives = 0
negatives = 0
for category in CLASSES:
    directory_path = os.path.join('data/val', category)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        image = load_image(os.path.join(directory_path, filename))
        pred = predict(image)
        if category == 'SCC':
            positives += 1
            true_positives += pred == 0
            false_negatives += pred != 0
        else:
            negatives += 1
            false_positives += pred == 0
            true_negatives += pred != 0
print('few-shot F1 score: ' + str(f1(true_positives, false_positives, false_negatives)))
print('few-shot accuracy: ' + str(accuracy(true_positives, true_negatives, false_positives, false_negatives)))
print('few-shot balanced accuracy: ' + str(balanced_accuracy(true_positives, true_negatives, positives, negatives)))