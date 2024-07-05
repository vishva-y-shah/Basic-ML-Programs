import streamlit as st
import pandas as pd

# Load the medical data CSV file
df = pd.read_csv('medical data.csv')

# Fill missing values in the 'Symptoms' column with an empty string
df['Symptoms'] = df['Symptoms'].fillna('')

def predict_disease_and_medicine():
    # Get user input
    symptoms = st.text_input("Enter your symptoms (comma-separated):")
    if symptoms:
        symptoms_list = symptoms.split(',')
        # Search the dataset for matching symptoms
        matching_records = df[df['Symptoms'].str.contains('|'.join(symptoms_list), case=False)]
        if not matching_records.empty:
            # Get the predicted disease and recommended medicine
            predicted_disease = matching_records['Symptoms'].iloc[0]
            recommended_medicine = matching_records['Medicine'].iloc[0]
            # Display the results
            st.write(f"Predicted disease: {predicted_disease}")
            st.write(f"Recommended medicine: {recommended_medicine}")
        else:
            st.write("No match found. Try different symptoms.")

# Create the Streamlit app
st.title("Disease Prediction and Medicine Recommendation")
st.write("Enter your symptoms and we'll try to predict the disease and recommend the appropriate medicine.")

predict_disease_and_medicine()
