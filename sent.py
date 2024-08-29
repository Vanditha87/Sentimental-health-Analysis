import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Display title
image_path = r"C:\Users\91703\Streamlit\Innomatics-Logo.jpeg"  # Replace with your actual PNG image file path
st.image(image_path)

# Display the PNG image
st.title("Sentiment Analysis for Mental Health  ")
st.write('This app performs sentiment analysis on mental health statements.')


try:
    model = pickle.load(open(r"C:\Users\91703\Streamlit\mh.pkl",'rb'))
    bow = pickle.load(open(r"C:\Users\91703\Streamlit\bow1.pkl",'rb'))
except Exception as e:
    st.error(f"An error occurred: {e}")


# Define the status labels (Ensure this matches the output labels of your model)
status_labels = {
    0: "Normal",
    1: "Stress",
    2: "Depression",
    3: "Anxiety",
    4: "Bipolar",
    5: "Personality disorder",
    6: "Suicidal"
}


# Text input for user statement
user_input = st.text_area("Enter a mental health statement:", "")

if user_input:
    # Transform the user input using the loaded CountVectorizer
    user_input_transformed = bow.transform([user_input])
    
    # Predict the sentiment using the loaded model
    prediction = model.predict(user_input_transformed)[0]
    
    # Display the result
    st.write(f"The predicted mental health status is: {prediction}")

st.button("Submit")