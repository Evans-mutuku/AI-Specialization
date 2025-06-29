import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    # The Flatten layer converts the 2D 28x28 images into a 1D array of 784 pixels.
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # A Dense (fully-connected) layer with 128 neurons and ReLU activation for introducing non-linearity.
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout layer randomly sets 20% of its inputs to zero during training.
    # This prevents overfitting by ensuring that the model does not rely too heavily on any particular set of features.
    tf.keras.layers.Dropout(0.2),
    
    # The final Dense layer with 10 neurons and softmax activation.
    # Each neuron corresponds to one of the 10 digits (0-9), and softmax outputs a probability distribution.
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model.
# - optimizer='adam': Adam optimizer adjusts the learning rate during training.
# - loss='sparse_categorical_crossentropy': This loss function is used for integer-labeled classification.
# - metrics=['accuracy']: The model will report accuracy during training and testing.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train (fit) the model on the training data over 5 epochs.
# An epoch means one full pass through the entire training dataset.
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test set.
# This provides an unbiased evaluation of how well the model generalizes to new, unseen data.
model.evaluate(x_test, y_test)