import os

import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from sklearn.metrics.pairwise import cosine_similarity
from utils import accuracy, f1, balanced_accuracy

IMAGE_SIZE = 128
BATCH_SIZE = 64
ONE_SHOT_BASE_SAMPLE_NAME = 'ISIC_0024329.jpg'
TEST = True


model = load_model('models/7-classes.h5')
model_cut = Model(inputs=model.input, outputs=model.layers[-2].output)

def load_image(file_path):
    img = tf.keras.utils.load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def collect_class_average(class_name):
    feature_samples = []
    sample_dir = os.path.join('data/train', class_name)
    for filename in os.listdir(sample_dir):
        image = load_image(os.path.join(sample_dir, filename))
        features = model_cut.predict(image, verbose=0)
        feature_samples.append(features[0])
    feature_samples = np.array(feature_samples)
    return np.average(feature_samples, axis=0).tolist()

if TEST:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    positives = 0
    negatives = 0
    features_averages = np.load('models/high_level_features.npy')
    for category in ['SCC', 'AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']:
        directory_path = os.path.join('data/test', category)
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            image = load_image(os.path.join(directory_path, filename))
            features = model_cut.predict(image, verbose=0)
            similarities = cosine_similarity(features_averages, features)
            if category == 'SCC':
                positives += 1
                true_positives += np.argmax(similarities) == 0
                false_negatives += np.argmax(similarities) != 0
            else:
                negatives += 1
                false_positives += np.argmax(similarities) == 0
                true_negatives += np.argmax(similarities) != 0
    print('one-shot F1 score: ' + str(f1(true_positives, false_positives, false_negatives)))
    print('one-shot accuracy: ' + str(accuracy(true_positives, true_negatives, false_positives, false_negatives)))
    print('one-shot balanced accuracy: ' + str(balanced_accuracy(true_positives, true_negatives, positives, negatives)))
else:
    features_averages = np.array([
        model_cut.predict(load_image('data/train/SCC/' + ONE_SHOT_BASE_SAMPLE_NAME), verbose=0)[0].tolist(),
        collect_class_average("AK"),
        collect_class_average("BCC"),
        collect_class_average("BKL"),
        collect_class_average("DF"),
        collect_class_average("MEL"),
        collect_class_average("NV"),
        collect_class_average("VASC")
    ])
    np.save('models/high_level_features.npy', features_averages)