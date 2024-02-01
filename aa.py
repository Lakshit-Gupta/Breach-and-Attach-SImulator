import gradio as gr
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Load the trained sklearn models
model0 = joblib.load('model.joblib')
model1 = joblib.load('sql New.joblib')
model2 = joblib.load('DDos New.joblib')

# Define preprocessing and postprocessing functions
def preprocess_data(input_data):
    # Implement your preprocessing logic here
    # This function should return the preprocessed data
    return input_data

def postprocess_prediction(prediction):
    # Implement your postprocessing logic here
    # This function should return the final prediction result
    return prediction

def on_button_click():
    # Do something when the button is clicked
    print("Kill Switch button clicked!")

def predict(input_data):
    # Convert input_data to a DataFrame
    input_data = pd.DataFrame([input_data], columns=['pktcount', 'byteperflow', 'tot_kbps', 'flows', 'bytecount', 'tot_dur'])

    # Preprocess the input data
    input_data = preprocess_data(input_data)

    # Make predictions using the loaded models
    prediction0 = model0.predict(input_data)
    prediction1 = model1.predict(input_data)
    prediction2 = model2.predict(input_data)

    # Postprocess the predictions
    prediction0 = postprocess_prediction(prediction0)
    prediction1 = postprocess_prediction(prediction1)
    prediction2 = postprocess_prediction(prediction2)

    return prediction0, prediction1, prediction2

# Create a custom HTML button
kill_switch_button = gr.HTML("<button onclick='killSwitchClick()'>Kill Switch</button>")

# Create Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="pktcount"),
        gr.Number(label="byteperflow"),
        gr.Number(label="tot_kbps"),
        gr.Number(label="flows"),
        gr.Number(label="bytecount"),
        gr.Number(label="tot_dur"),
        kill_switch_button,  # Use the custom HTML button
    ],
    outputs=[
        gr.Textbox(label="Prediction 1", output_id="prediction_0"),
        gr.Textbox(label="Prediction 2", output_id="prediction_1"),
        gr.Textbox(label="Prediction 3", output_id="prediction_2"),
    ]
)

# Launch the Gradio interface
iface.launch()
