import os

import numpy as np
from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
import tensorflow as tf
from utils import accuracy, f1, balanced_accuracy

IMAGE_SIZE = 128
BATCH_SIZE = 64
TRAINING_EPOCHS = 10
LEARNING_RATE = 0.001
CLASS_COUNT = 8
TEST = True


model = Sequential([
    ResNet50(include_top = False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    GlobalAveragePooling2D(),
    Dense(CLASS_COUNT, activation = 'softmax')
])
model.layers[0].trainable = False
model.compile(optimizer = Adam(learning_rate=LEARNING_RATE), loss = 'categorical_crossentropy', metrics = ['accuracy'])

training_dataset = (tf.keras.utils.image_dataset_from_directory('data/train' if CLASS_COUNT == 8 else 'data/train-excluded', image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE)
                    .map(lambda x, y: (x, tf.one_hot(y, depth=CLASS_COUNT))))
validation_dataset = (tf.keras.utils.image_dataset_from_directory('data/val' if CLASS_COUNT == 8 else 'data/val-excluded', image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE)
                      .map(lambda x, y: (x, tf.one_hot(y, depth=CLASS_COUNT))))

def load_image(file_path):
    img = tf.keras.utils.load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


if TEST:
    model.load_weights('models/{0}-classes.h5'.format(CLASS_COUNT))
    if CLASS_COUNT == 8:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        positives = 0
        negatives = 0
        for category in ['SCC', 'AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']:
            directory_path = os.path.join('data/test', category)
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                probs = model.predict(load_image(file_path), verbose=0)
                if category == 'SCC':
                    positives += 1
                    true_positives += np.argmax(probs[0]) == 6
                    false_negatives += np.argmax(probs[0]) != 6
                else:
                    negatives += 1
                    true_negatives += np.argmax(probs[0]) != 6
                    false_positives += np.argmax(probs[0]) == 6
        print('full-data F1 score: ' + str(f1(true_positives, false_positives, false_negatives)))
        print('full-data accuracy: ' + str(accuracy(true_positives, true_negatives, false_positives, false_negatives)))
        print('full-data balanced accuracy: ' + str(balanced_accuracy(true_positives, true_negatives, positives, negatives)))
    else:
        model.evaluate(validation_dataset)
else:
    model.fit(training_dataset, validation_data=validation_dataset, epochs=TRAINING_EPOCHS)
    model.save('models/{0}-classes.h5'.format(CLASS_COUNT))