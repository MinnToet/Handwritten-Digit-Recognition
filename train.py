import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import cv2
import os
import time

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
# Train the model
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=100,
    validation_data=(x_test, y_test)
)

training_time = time.time() - start_time
# Evaluate the model 
# Evaluation 

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

y_pred = model.predict(x_test, verbose=0)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

precision = precision_score(
    y_true,
    y_pred_class,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred_class,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred_class,
    average="weighted"
)

cm = confusion_matrix(
    y_true,
    y_pred_class
)

report = classification_report(
    y_true,
    y_pred_class
)

print("="*50)
print("Evaluation")
print("="*50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Loss     : {loss:.4f}")
print(f"Training Time : {training_time:.2f} seconds")


os.makedirs("results", exist_ok=True)

with open("results/train_v1_metrics.txt", "w") as f:

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"Loss     : {loss:.4f}\n")
    f.write(f"Training Time : {training_time:.2f} seconds\n\n")

    f.write("Classification Report\n")
    f.write(report)

    f.write("\n\nConfusion Matrix\n")
    f.write(np.array2string(cm))
model.save('handwritten.h5')    # we will save our model with name : mnist.h5

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()

plt.tight_layout()

plt.savefig("results/train_v1_history.png", dpi=300)

plt.show()