import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
data = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print('Training the model...')
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
print('Evaluating the model...')
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Test on a random image from the test set
def test_random_image():
    idx = np.random.randint(0, len(x_test))
    img = x_test[idx]
    plt.imshow(img, cmap='gray')
    plt.title('Sample Image')
    plt.show()
    img_reshaped = img.reshape(1, 28, 28)
    prediction = np.argmax(model.predict(img_reshaped), axis=1)
    print(f'Predicted label: {prediction[0]}')
    print(f'Actual label: {y_test[idx]}')

if __name__ == '__main__':
    test_random_image() 