from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
import tensorflow as tf

IMAGE_SIZE = 128
BATCH_SIZE = 64
TRAINING_EPOCHS = 10
LEARNING_RATE = 0.001
CLASS_COUNT = 7

model = Sequential([
    ResNet50(include_top = False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    GlobalAveragePooling2D(),
    Dense(CLASS_COUNT, activation = 'softmax')
])
model.layers[0].trainable = False
model.compile(optimizer = Adam(learning_rate=LEARNING_RATE), loss = 'categorical_crossentropy', metrics = ['accuracy'])

training_dataset = (tf.keras.utils.image_dataset_from_directory('data/train', image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE)
                    .map(lambda x, y: (x, tf.one_hot(y, depth=CLASS_COUNT))))
validation_dataset = (tf.keras.utils.image_dataset_from_directory('data/val', image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE)
                      .map(lambda x, y: (x, tf.one_hot(y, depth=CLASS_COUNT))))

model.fit(training_dataset, validation_data=validation_dataset, epochs=TRAINING_EPOCHS)

model.save('models/base-7-classes.h5')