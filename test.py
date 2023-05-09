import time

import streamlit as st
import pandas as pd
from PIL import Image
import pickle

st.set_page_config(page_title='dd', page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.header('WerkOps')
col1, col2, col3 = st.columns(3)

with col1:
    image = Image.open(r"C:\Users\heram\PycharmProjects\DecisionTree_BMWChallenge\try-dt.PNG")

    st.subheader('Framework Decision for AI')
    st.image(image, caption='Radial Version')
    # st.image(r"C:\Users\heram\PycharmProjects\DecisionTree_BMWChallenge\Capture.PNG", format="PNG")

# Load the trained model from a pickle file
with open(r'C:\Users\heram\PycharmProjects\DecisionTree_BMWChallenge\dt_model.pkl', 'rb') as file:
    model = pickle.load(file)

with col2:
    # Add a title to the app
    st.subheader("Decision Parameters")

    col1, col2 = st.columns(2)
    with col1:
        # Add a slider for selecting a value between 0 and 100
        slider1_val = st.slider(":blue[Batch Size]", 0, 100)

        # Add a slider for selecting a value between 0 and 100
        slider2_val = st.slider(":blue[Latency (ms)]", 0.0, 12000.00)

        # Add a slider for selecting a value between 0 and 100
        slider3_val = st.slider(":blue[Throughput (fps)]", 0.0, 200.00)
    with col2:
        # Add a dropdown for selecting a color
        model_options = ['vit_b_16', 'vgg11', 'resnet50']
        model_val = st.selectbox(":blue[Model]", model_options)

        # Add a dropdown for selecting a color
        precision_options = ['FP16', 'FP32']
        precision_val = st.selectbox(":blue[Precision]", precision_options)

with col3:
    st.subheader("Selected Parameters")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(":blue[Batch Size]", value=slider1_val)
    st.metric(":blue[Latency]", value=slider2_val)
    col3.metric(":blue[Throughput]", value=slider3_val)
    st.metric(":blue[Model]", value=model_val)
    st.metric(":blue[Precision]", value=precision_val)

    # Use the model to make a prediction based on the slider and dropdown values
    start_time = time.time()
    prediction = model.predict([[slider1_val, slider2_val, slider3_val,
                                 model_options.index(model_val),
                                 precision_options.index(precision_val)]])[0]
    end_time = time.time()

    # Display the predicted value and prediction time
    st.subheader("Decision :")
    col1, col2, col3 = st.columns(3)
    with col1:
        if prediction[0] == 0:
            st.metric(":blue[Processor]", "CPU")
        elif prediction[0] == 1:
            st.metric(":blue[Processor]", "GPU")
    with col2:
        if prediction[1] == 0:
            st.metric(":blue[Machine]", "Azure")
        elif prediction[1] == 1:
            st.metric(":blue[Machine]", "Xeon")
        elif prediction[1] == 2:
            st.metric(":blue[Machine]", "i5")
        elif prediction[1] == 3:
            st.metric(":blue[Machine]", "i7")
        elif prediction[1] == 4:
            st.metric(":blue[Machine]", "i9")

    with col3:
        st.metric(f":blue[Prediction time (sec):]", f"{end_time - start_time:.5f}")
