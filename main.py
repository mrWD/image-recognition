import numpy as np
import keras
from keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import matplotlib.pyplot as plt

num_classes = 10  # Number of output classes (digits 0-9)
input_shape = (28, 28, 1)  # Shape of each input image (28x28 pixels, 1 color channel)

# Load the MNIST dataset and split it into training and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # x_train and x_test are images, y_train and y_test are labels

# Normalize the data - scale pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255  # Convert training images to float32 and divide by 255
x_test = x_test.astype("float32") / 255  # Convert test images to float32 and divide by 255

# Change image shape from (28, 28) to (28, 28, 1), because Keras expects a 3D input for Conv2D
x_train = np.expand_dims(x_train, -1)  # Add a new axis at the end for the color channel
x_test = np.expand_dims(x_test, -1)  # Add a new axis at the end for the color channel
print("x_train shape:", x_train.shape)  # Print the shape of the training data
print(x_train.shape[0], "train samples")  # Print the number of training samples
print(x_test.shape[0], "test samples")  # Print the number of test samples

# Convert labels to one-hot encoded vectors: e.g., 5 -> 0000100000
y_train_cat = keras.utils.to_categorical(y_train, num_classes)  # Convert training labels to one-hot encoding
y_test_cat = keras.utils.to_categorical(y_test, num_classes)  # Convert test labels to one-hot encoding

# Build the neural network model using Sequential API
# Conv2D(num_filters, kernel_size), MaxPooling2D(pool_size)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),  # Input layer specifying the shape of input images
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),  # Convolutional layer with 32 filters and 3x3 kernel
        layers.MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer with 2x2 pool size
        #layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),  # Another convolutional layer
        #layers.MaxPooling2D(pool_size=(2, 2)),  # Another max pooling layer
        layers.Flatten(),  # Flatten the 3D output to 1D for the dense layer
        #layers.Dropout(0.5), # Dropout layer to prevent overfitting
        #Dense(128, activation='relu'),  # Dense layer with 128 units and ReLU activation
        layers.Dense(num_classes, activation="softmax"),  # Output layer with softmax activation for classification
    ]
)

model.summary()  # Print a summary of the model architecture

batch_size = 128  # Number of samples per gradient update
epochs = 15  # Number of times to iterate over the training data

# Compile the model: specify loss function, optimizer, and metrics to monitor
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model on the training data
model.fit(x_train, y_train_cat, batch_size=batch_size, epochs=epochs, validation_split=0.1)  # Use 10% of training data for validation

# Evaluate the model on the test data
score = model.evaluate(x_test, y_test_cat, verbose=0)  # Returns loss and accuracy
print("Test loss:", score[0])  # Print the test loss
print("Test accuracy:", score[1])  # Print the test accuracy

import matplotlib.pyplot as plt  # Import matplotlib again (already imported above)
all = model.predict(x_test)  # Predict the classes for the test images
all = np.argmax(all, axis=1)  # Convert predictions from one-hot to class labels
true = all == y_test  # Boolean array: True if prediction matches true label
x_false = x_test[~true]  # Select images where prediction was wrong
y_false = y_test[~true]  # Select true labels for wrong predictions
predsk = all[~true]  # Select predicted labels for wrong predictions
print(y_false[9])  # Print the true label of the 10th misclassified image
print ('Predsk:')  # Print label
print(predsk[9])  # Print the predicted label of the 10th misclassified image
plt.imshow(x_false[9], cmap=plt.cm.binary)  # Show the 10th misclassified image in grayscale
