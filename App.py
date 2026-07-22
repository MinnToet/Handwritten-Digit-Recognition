import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import time 

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten_v3.h5")

model = load_model()
 

def predictDigit(image):

    image = ImageOps.grayscale(image)

    img = image.resize((28, 28))

    img = np.array(img, dtype="float32") / 255.0

    img = img.reshape((1, 28, 28, 1))

    # Warm-up (không tính vào thời gian)
    model.predict(img, verbose=0)

    start = time.perf_counter()

    pred = model.predict(img, verbose=0)

    end = time.perf_counter()

    inference_time = (end - start) * 1000  # ms

    digit = np.argmax(pred[0])

    confidence = float(pred[0][digit] * 100)

    return digit, confidence, pred[0], inference_time

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    layout="wide"
)

st.title("Handwritten Digit Recognition")

st.write("Draw a digit and click **Predict**")

col1, col2 = st.columns([1, 1])
 

with col1:

    stroke_width = st.slider(
        "Stroke Width",
        1,
        30,
        15
    )

    canvas_result = st_canvas(

        stroke_width=stroke_width,

        stroke_color="#FFFFFF",

        background_color="#000000",

        fill_color="rgba(255,165,0,0.3)",

        height=280,

        width=280,

        drawing_mode="freedraw",

        key="canvas"

    )

    predict_btn = st.button("Predict")
 
with col2:

    if predict_btn:

        if canvas_result.image_data is not None:

            input_numpy_array = np.array(canvas_result.image_data)

            input_image = Image.fromarray(
                input_numpy_array.astype("uint8"),
                "RGBA"
            )

            digit, confidence, prob, inference_time = predictDigit(input_image)

            st.markdown("## Prediction")

            st.metric(
                label="Predicted Digit",
                value=str(digit)
            )

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

            st.metric(
                label="Inference Time",
                value=f"{inference_time:.3f} ms"
            )
            st.markdown("### Probability Distribution")

            df = pd.DataFrame({

                "Digit": [str(i) for i in range(10)],

                "Probability (%)": prob * 100

            })

            st.bar_chart(
                df,
                x="Digit",
                y="Probability (%)"
            )

        else:

            st.warning("Please draw a digit.")

