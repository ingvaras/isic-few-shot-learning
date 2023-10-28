from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential([
    ResNet50(include_top = False, weights='imagenet', input_shape=(256, 256, 3)),
    Dense(8, activation = 'softmax')
])
model.layers[0].trainable = False
model.compile(optimizer = Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(model.summary())
