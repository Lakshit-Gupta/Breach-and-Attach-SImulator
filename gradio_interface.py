import gradio as gr
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression

# Load the trained sklearn model
model0 = joblib.load('model.joblib')
model1 = joblib.load('sql New.joblib')
model2= joblib.load('DDos New.joblib')
#input0_data =

# Define preprocessing and postprocessing functions
def preprocess_data(input_data):
    # Implement your preprocessing logic here
    # This function should return the preprocessed data
    return input_data


def postprocess_prediction(prediction):
    # Implement your postprocessing logic here
    # This function should return the final prediction result
    return prediction

def On_button_click(event):
    # Do something when the button is clicked
    print("Kill Switch button clicked!")
def predict(input_data):
    # Convert input_data to a DataFrame
    input_data = pd.DataFrame([input_data], columns=['pktcount', 'byteperflow', 'tot_kbps', 'flows','bytecount','tot_dur'])
    input_data = pd.DataFrame([input_data],
                              columns=['pktcount', 'byteperflow', 'tot_kbps', 'flows', 'bytecount', 'tot_dur'])
    input_data = pd.DataFrame([input_data],
                              columns=['pktcount', 'byteperflow', 'tot_kbps', 'flows', 'bytecount', 'tot_dur'])

    # Preprocess the input data
    input_data = preprocess_data(input_data)

    # Make a prediction using the loaded model
    prediction = model0.predict(input_data)

    # Postprocess the prediction
    prediction = postprocess_prediction(prediction)

    return prediction

# Create
# iface = gr.Interface(
#     fn=predict,
#     inputs=gr.input_type.TextboxGroup([gr.input_type.Number() for _ in range(4)], label="Input Features"),
#     outputs="text"
# )
iface = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label="status"),
            ],
    outputs=[gr.Number(label="logs"),
             gr.Number(label="ddos attack "),
             gr.Number(label="ransomware"),
             gr.Number(label="sql enjection attack"),
             gr.Number(label="phishing   attack"),])
 # )
#iface.add(gr.Button("Kill Switch"))

# Launch the Gradio interface
iface.launch()
