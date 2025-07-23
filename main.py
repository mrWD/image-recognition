# Import numpy for numerical operations
import numpy as np
# Import matplotlib for plotting images
import matplotlib.pyplot as plt

# Import MNIST dataset and layers from Keras
from keras.datasets import mnist
from tensorflow import keras
from keras.layers import Dense, Flatten

# Load the MNIST dataset (handwritten digits)
# x_train, y_train: training images and labels
# x_test, y_test: test images and labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print a label from the training set (uncomment to use)
# print(y_train[12])
# Print the image data as a matrix (uncomment to use)
# print(x_train[12])

# Display the image using matplotlib (uncomment to use)
# plt.imshow(x_train[12], cmap='gray')

# Normalize the image data to be between 0 and 1
x_train = x_train / 255
x_test = x_test / 255
# print(x_train[12])  # Check the normalized data (uncomment to use)

# Convert labels to one-hot encoded vectors (e.g., 3 -> [0,0,0,1,0,0,0,0,0,0])
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Build a simple neural network model
model = keras.Sequential([
    # Flatten the 28x28 image into a 784-length vector
    Flatten(input_shape=(28, 28, 1)),
    # Add a dense (fully connected) layer with 20 neurons and ReLU activation
    Dense(20, activation='relu'),
    # Output layer: 10 neurons (one for each digit), softmax for probabilities
    Dense(10, activation='softmax')
])

# Print a summary of the model (uncomment to use)
# print(model.summary())

# Compile the model: set optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model on the training data
# batch_size=32: update weights after 32 samples
# epochs=5: go through the data 5 times
# validation_split=0.2: use 20% of training data for validation
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
# Evaluate the model on the test data
model.evaluate(x_test, y_test_cat)

# Prepare a single test image for prediction (add batch dimension)
x = np.expand_dims(x_test[12], axis=0)
# Predict the probabilities for each digit
result = model.predict(x)
# print(result)  # Show raw probabilities (uncomment to use)
# print(np.argmax(result))  # Show predicted digit (uncomment to use)

# plt.imshow(x_test[12], cmap='gray')  # Show the test image (uncomment to use)

# Predict all test images
all = model.predict(x_test)
# Convert probabilities to digit predictions
all = np.argmax(all, axis=1)
# print(all[:20])  # Show first 20 predictions (uncomment to use)
# print(y_test[:20])  # Show first 20 true labels (uncomment to use)

# Find which predictions are correct (True) or incorrect (False)
true = all == y_test
# print(true[:10])  # Show first 10 correctness values (uncomment to use)

# print(x_false.shape)  # Check shape of incorrect images (uncomment to use)

# Select only the incorrectly predicted images and their labels
x_false = x_test[~true]
y_false = y_test[~true]
predsk = all[~true]

# Print the true label of the 11th incorrect prediction
print(y_false[10])
print ('Predsk:')
# Print the predicted label for the same image
print(predsk[10])

# Show the 11th incorrectly predicted image
plt.imshow(x_false[10], cmap=plt.cm.binary)
