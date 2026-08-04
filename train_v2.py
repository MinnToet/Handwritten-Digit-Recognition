import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

NUM_CLASSES = 10
CLASS_LABELS = list(range(NUM_CLASSES))

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)
 
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

x_train_full = np.expand_dims(x_train_full, axis=-1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype("float32") / 255.0
 
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=0.1,
    random_state=SEED,
    stratify=y_train_full
)

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat = to_categorical(y_val, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)

print("Train:", x_train.shape)
print("Val  :", x_val.shape)
print("Test :", x_test.shape)

datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.10
) 
datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(NUM_CLASSES, activation="softmax"))
model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
 
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "model/handwritten_v2.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

start_time = time.time()
history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=32, seed=SEED),
    epochs=100,
    validation_data=(x_val, y_val_cat),
    callbacks=callbacks,
    verbose=1
)
training_time = time.time() - start_time
 
loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)

y_pred = model.predict(x_test, verbose=0)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

precision = precision_score(y_true, y_pred_class, average="weighted",
                             labels=CLASS_LABELS, zero_division=0)
recall = recall_score(y_true, y_pred_class, average="weighted",
                       labels=CLASS_LABELS, zero_division=0)
f1 = f1_score(y_true, y_pred_class, average="weighted",
              labels=CLASS_LABELS, zero_division=0)
cm = confusion_matrix(y_true, y_pred_class, labels=CLASS_LABELS)
report = classification_report(y_true, y_pred_class, labels=CLASS_LABELS,
                                digits=4, zero_division=0)

print("=" * 50)
print("Evaluation (train_v2.py)")
print("=" * 50)
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"Loss          : {loss:.4f}")
print(f"Training Time : {training_time:.2f} seconds")

with open("results/train_v2_metrics.txt", "w") as f:
    f.write(f"Accuracy      : {accuracy:.4f}\n")
    f.write(f"Precision     : {precision:.4f}\n")
    f.write(f"Recall        : {recall:.4f}\n")
    f.write(f"F1 Score      : {f1:.4f}\n")
    f.write(f"Loss          : {loss:.4f}\n")
    f.write(f"Training Time : {training_time:.2f} seconds\n\n")
    f.write("Classification Report\n")
    f.write(report)
    f.write("\n\nConfusion Matrix\n")
    f.write(np.array2string(cm))
 
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("results/train_v2_history.png", dpi=300)
plt.show()