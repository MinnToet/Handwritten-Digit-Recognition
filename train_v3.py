import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

import time
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

def rotate_image(img):

    angle = random.uniform(-15, 15)

    h, w = img.shape

    M = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderValue=0
    )

def translate_image(img):

    tx = random.randint(-3, 3)
    ty = random.randint(-3, 3)

    M = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    return cv2.warpAffine(
        img,
        M,
        (28, 28),
        borderValue=0
    )

def scale_image(img):

    scale = random.uniform(0.85, 1.15)

    h, w = img.shape

    M = cv2.getRotationMatrix2D(
        (w//2, h//2),
        0,
        scale
    )

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderValue=0
    )

def dilate_image(img):

    kernel = np.ones((2,2), np.uint8)

    return cv2.dilate(
        img,
        kernel,
        iterations=1
    )

def erode_image(img):

    kernel = np.ones((2,2), np.uint8)

    return cv2.erode(
        img,
        kernel,
        iterations=1
    )

def blur_image(img):

    return cv2.GaussianBlur(
        img,
        (3,3),
        0
    )

def augment(img):

    image = img.copy()

    if random.random() < 0.6:
        image = rotate_image(image)

    if random.random() < 0.6:
        image = translate_image(image)

    if random.random() < 0.5:
        image = scale_image(image)

    if random.random() < 0.3:
        image = dilate_image(image)

    if random.random() < 0.3:
        image = erode_image(image)

    if random.random() < 0.3:
        image = blur_image(image)

    return image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Original training:", x_train.shape)

new_images = []
new_labels = []

for img, label in zip(x_train, y_train):

    new_images.append(img)
    new_labels.append(label)

    for _ in range(2):

        aug = augment(img)

        new_images.append(aug)

        new_labels.append(label)

x_train = np.array(new_images)

y_train = np.array(new_labels)

print("Augmented:", x_train.shape)
 
# Normalize 

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print("Train:", x_train.shape)
print("Test :", x_test.shape)
 
# CNN 
model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(28,28,1)
    )
)

model.add(
    MaxPooling2D((2,2))
)

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(
    MaxPooling2D((2,2))
)

model.add(Flatten())

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dense(
        10,
        activation="softmax"
    )
)

model.summary()
 
# Compile 

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

 
# Train 
start_time = time.time()
history = model.fit(

    x_train,

    y_train,

    batch_size=128,

    epochs=100,

    validation_data=(

        x_test,

        y_test

    ),

    verbose=1

)
training_time = time.time() - start_time
 
# Evaluation 

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

y_pred = model.predict(
    x_test,
    verbose=0
)

y_pred_class = np.argmax(
    y_pred,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)

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

print("="*60)
print("Evaluation")
print("="*60)

print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"Loss          : {loss:.4f}")
print(f"Training Time : {training_time:.2f} seconds")
 
# Save metrics 

with open(
    "results/train_v3_metrics.txt",
    "w"
) as f:

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
 
# Save history 
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["Train","Validation"])

plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["Train","Validation"])

plt.tight_layout()

plt.savefig(
    "results/train_v3_history.png",
    dpi=300
)

plt.show()

# Save model

model.save("model/handwritten_v3.h5")