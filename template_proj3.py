from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
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
print(labels[0], labels[650], labels[1300])


#
#
#
# Fill in Model Here
#
#


# End Custom Code

# Add last layer and "activate"
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val), 
                    epochs=10, 
                    batch_size=512)


# Report Results

print(history.history)
model.predict()