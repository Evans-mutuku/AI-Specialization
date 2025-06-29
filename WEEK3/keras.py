import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using the training data.
# The model will go through the data for 5 epochs (complete passes through the training data).
model.fit(x_train, y_train, epochs=5)

# Evaluate the model performance on the test data.
# This step computes the model's loss and accuracy on the unseen test dataset.
model.evaluate(x_test, y_test)
