from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from random import randrange
import numpy as np

# Model Template

model = Sequential()  # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))

"""
Custom Code Below
"""

# Import images and labels files
images = np.load("images.npy")
labels = np.load("labels.npy")

np.reshape(images, (6500, 784))  # Turn image matrices into vectors
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)  # Convert label integers to one-hot encodings

# labels are organized in order from 0 -> 9, so the 650th index is the first '1' label
# print(labels[0], labels[650], labels[1300])

# Separate Data
# images_0 = np.ndarray(shape=(650, 784))
#
# for i in range(0, 650):
#     images_0.put(i, images[i])

# print(images_0)

#
#
# OFFICE HOURS PSEUDOCODE
#
#

# Generate random var between 0 and 100 for each index, if >=60 for example, put BOTH image and label into training set
# Training Set: 60%
training_images = np.ndarray(shape=(650, 784))
training_labels = np.ndarray(shape=(650, 784))
# Validation Set: 15%
validation_images = np.ndarray(shape=(650, 784))
validation_labels = np.ndarray(shape=(650, 784))
# Test Set: 25%
test_images = np.ndarray(shape=(650, 784))
test_labels = np.ndarray(shape=(650, 784))

for i in range(0, 6500):
    rand = randrange(0, 100)
    if rand < 60:
        # put it in the training set
        training_images.put(i, images[i])
        training_labels.put(i, one_hot_labels[i])
    elif 60 <= rand < 75:
        # put it in the validation set
        validation_images.put(i, images[i])
        validation_labels.put(i, one_hot_labels[i])
    else:
        # put it in the test set
        test_images.put(i, images[i])
        test_labels.put(i, one_hot_labels[i])

#
#
#
# Fill in Model Here
#
#
model.add(Dense(10))


"""
End Custom Code
"""
# Add last layer and "activate"
model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
model.add(Activation('softmax'))

print("Compiling the model")
# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("Training the model")
# Train Model
history = model.fit(training_images, training_labels,
                    validation_data = (validation_images, validation_labels),
                    epochs=10, 
                    batch_size=512)


# Report Results

print(history.history)
model.predict()