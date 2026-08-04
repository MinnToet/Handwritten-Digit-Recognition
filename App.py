import time

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

MODEL_PATH = "model/handwritten_v2.h5"


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except (OSError, ValueError) as exc:
        st.error(f"Không tải được model tại '{MODEL_PATH}': {exc}")
        st.stop()

    # FIX: warm-up chỉ chạy đúng 1 lần ở đây (nhờ @st.cache_resource),
    # thay vì chạy lại mỗi lần người dùng bấm "Predict" như code cũ -
    # trước đây mỗi lần predict thực chất gọi model.predict() 2 lần.
    model.predict(np.zeros((1, 28, 28, 1), dtype="float32"), verbose=0)
    return model


model = load_model()


def predict_digit(image: Image.Image):
    image = ImageOps.grayscale(image)

    # FIX: chỉ định rõ filter LANCZOS cho resize, để chất lượng downsize
    # nhất quán giữa các phiên bản Pillow khác nhau (trước đây dùng filter
    # mặc định, có thể đổi giữa các bản Pillow).
    img = image.resize((28, 28), Image.LANCZOS)

    img = np.array(img, dtype="float32") / 255.0
    img = img.reshape((1, 28, 28, 1))

    start = time.perf_counter()
    pred = model.predict(img, verbose=0)
    end = time.perf_counter()

    inference_time = (end - start) * 1000  # ms

    digit = int(np.argmax(pred[0]))
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
    stroke_width = st.slider("Stroke Width", 1, 30, 15)

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

            digit, confidence, prob, inference_time = predict_digit(input_image)

            st.markdown("## Prediction")

            st.metric(label="Predicted Digit", value=str(digit))
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            st.metric(label="Inference Time", value=f"{inference_time:.3f} ms")

            st.markdown("### Probability Distribution")

            df = pd.DataFrame({
                "Digit": [str(i) for i in range(10)],
                "Probability (%)": prob * 100
            })

            st.bar_chart(df, x="Digit", y="Probability (%)")
        else:
            st.warning("Please draw a digit.")