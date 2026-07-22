import os
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
 
import time

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
 

os.makedirs("model", exist_ok=True)
 

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("Train:", x_train.shape)
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
 

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)
 

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "model/handwritten_v2.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

]

start_time = time.time()

history = model.fit(

    datagen.flow(
        x_train,
        y_train,
        batch_size=32
    ),

    epochs=100,

    validation_data=(
        x_test,
        y_test
    ),

    callbacks=callbacks,

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

print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"Loss          : {loss:.4f}")
print(f"Training Time : {training_time:.2f} seconds")


os.makedirs("results", exist_ok=True)

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

plt.savefig(
    "results/train_v2_history.png",
    dpi=300
)

plt.show()