from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from random import randrange
from sklearn.metrics import confusion_matrix
import numpy as np


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = ' ', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

# Model Template

model = Sequential()  # declare model
model.add(Dense(10, input_shape=(28 * 28,), kernel_initializer='he_normal'))  # first layer
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

# Generate random var between 0 and 100 for each index, if >=60 for example, put BOTH image and label into training set
# Training Set: 60%
training_images = np.zeros(shape=(3900, 784))
training_labels = np.zeros(shape=(3900, 10))
t_i_l = []
t_l_l = []
# Validation Set: 15%
validation_images = np.zeros(shape=(975, 784))
validation_labels = np.zeros(shape=(975, 10))
v_i_l = []
v_l_l = []

# Test Set: 25%
test_images = np.zeros(shape=(1625, 784))
test_labels = np.zeros(shape=(1625, 10))
te_i_l = []
te_l_l = []

training_counter = 0
validation_counter = 0
test_counter = 0

# printProgressBar(0, 6500, prefix='Sampling:', suffix='Complete', length=50)
print("Starting Stratified Sampling")
for i in range(0, 6500):
    # printProgressBar(i + 1, 6500, prefix='Sampling:', suffix='Complete', length=50)
    rand = randrange(0, 100)
    if rand < 60:
        # put it in the training set
        t_i_l.append(images[i])
        t_l_l.append(one_hot_labels[i])
        # training_images = np.insert(training_images, training_counter, images[i], 0)
        # training_labels = np.insert(training_labels, training_counter, one_hot_labels[i], 0)
        # training_counter += 1
    elif 60 <= rand < 75:
        # put it in the validation set
        # validation_images = np.insert(validation_images, validation_counter, images[i], 0)
        # validation_labels = np.insert(validation_labels, validation_counter, one_hot_labels[i], 0)
        # validation_counter += 1
        v_i_l.append(images[i])
        v_l_l.append(one_hot_labels[i])
    else:
        # put it in the test set
        # test_images = np.insert(test_images, test_counter, images[i], 0)
        # test_labels = np.insert(test_labels, test_counter, one_hot_labels[i], 0)
        # test_counter += 1
        te_i_l.append(images[i])
        te_l_l.append(one_hot_labels[i])

print("Done Sampling")

training_images = np.array(t_i_l)
training_labels = np.array(t_l_l)

validation_images = np.array(v_i_l)
validation_labels = np.array(v_l_l)

test_images = np.array(te_i_l)
test_labels = np.array(te_l_l)

#
#
# Fill in Model Here
#
#
model.add(Dense(10))
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
                    validation_data=(validation_images, validation_labels),
                    epochs=10,
                    batch_size=512)

# Report Results

print(history.history)
predictions = model.predict(test_images)

print("Actual:", test_labels)
print("Predicted: ", predictions)

conf_matrix = confusion_matrix(test_labels, predictions, labels=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])

print(conf_matrix)
