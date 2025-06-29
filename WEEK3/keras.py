# Import TensorFlow, which includes the Keras API.
import tensorflow as tf

# Load the MNIST dataset provided by Keras.
# This function downloads the dataset (if not already present locally) and splits it
# into training and testing sets. The images are stored in x_train and x_test, and their corresponding labels are stored in y_train and y_test.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the image data.
# The pixel values in the MNIST images range from 0 to 255. Dividing by 255.0 scales these values to the range 0 to 1.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the neural network model using the Sequential API.
# The model is a linear stack of layers.
model = tf.keras.models.Sequential([
    # Flatten layer:
    # Converts each 28x28 image into a 1D array of 784 pixels.
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Dense (fully-connected) layer:
    # Has 128 neurons and uses the ReLU activation function to add non-linearity.
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout layer:
    # Randomly sets 20% of the input units to 0 at each update during training to reduce the risk of overfitting.
    tf.keras.layers.Dropout(0.2),
    
    # Output layer:
    # A Dense layer with 10 neurons—one for each digit (0-9)—and a softmax activation,
    # which converts the outputs into probabilities that sum to 1.
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model.
# - optimizer: The Adam optimizer adapts the learning rate during training.
# - loss: 'sparse_categorical_crossentropy' is appropriate for multi-class classification (when labels are provided as integers).
# - metrics: We track 'accuracy' during training and testing.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using the training data.
# The model will go through the data for 5 epochs (complete passes through the training data).
model.fit(x_train, y_train, epochs=5)

# Evaluate the model performance on the test data.
# This step computes the model's loss and accuracy on the unseen test dataset.
model.evaluate(x_test, y_test)
