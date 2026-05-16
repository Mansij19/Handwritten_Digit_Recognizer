"""
Train the handwritten digit classification CNN and save as digit_model.h5.
Uses Keras built-in MNIST dataset with enhanced training for better accuracy.
"""
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ─── Load Data ────────────────────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ─── Preprocess ───────────────────────────────────────────────────────
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)

# ─── Build Improved Model ────────────────────────────────────────────
cnn = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax"),
])

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ─── Data Augmentation (light) ────────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode="constant"
)

# ─── Train with Early Stopping ───────────────────────────────────────
early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
cnn.fit(datagen.flow(X_train, y_train_cat, batch_size=64),
        epochs=20,
        validation_data=(X_test, y_test_cat),
        callbacks=[early_stop],
        verbose=1)

# ─── Evaluate ─────────────────────────────────────────────────────────
loss, acc = cnn.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✓ Test Accuracy: {acc*100:.2f}%")

# ─── Save ─────────────────────────────────────────────────────────────
cnn.save("digit_model.h5")
print("✓ Model saved as digit_model.h5")
