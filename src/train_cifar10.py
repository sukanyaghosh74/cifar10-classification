import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=30, batch_size=64, callbacks=[early_stop])

# Save the trained model
model.save("models/cifar10_cnn.h5")
print("Model saved at models/cifar10_cnn.h5")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label="Train Accuracy", marker="o")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.savefig("results/training_plot.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label="Train Loss", marker="o", linestyle="dashed")
plt.plot(history.history['val_loss'], label="Validation Loss", marker="o", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig("results/loss_plot.png")
plt.show()

print("Training plots saved in results/")
