# -*- coding: utf-8 -*-
"""
Made by Nitin pal ~ 26/01/2024 ; 10:48:04 PM
"""

import streamlit as st
from PIL import Image
import base64
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import time
import random

# Load the model and preprocessing information
with open('data/random forest_model.pkl', 'rb') as model_file:
    saved_data = pickle.load(model_file)

# Extract model and preprocessing information
loaded_model = saved_data['model']
scaler = saved_data['scaler']
encoder = saved_data['encoder']
feature_selector = saved_data['feature_selector']


# Function for input validation
def validate_input(age, bmi, avg_glucose_level):
    if not 0 <= bmi <= 70 or not 0 <= avg_glucose_level <= 300:
        st.error("Invalid input. Please ensure BMI is between 0 and 70, and glucose level is between 0 and 300.")
        st.stop()

# Function to normalize the input data
@st.cache_data(hash_funcs={MinMaxScaler: hash})
def normalize_input_data(input_data, scaler=None):
    if scaler is None:
        return input_data
    return scaler.transform(input_data)

# Function to make predictions
@st.cache_data(hash_funcs={RandomForestClassifier: lambda _: None})
def predict_stroke_risk(normalized_data, model):
    prediction = model.predict(normalized_data)
    confidence = model.predict_proba(normalized_data)[:, 1]
    return prediction, confidence

# Function to handle age and work type consistency
def validate_age_work_type(age, work_type):
    if work_type == "Child":
        return max(age, 0)
    else:
        return max(age, 17)

# Function for categorical mapping
def map_categorical(value, mapping_dict):
    return mapping_dict[value]

def animated_line():
    st.markdown("""
        <style>
            .animated-line {
                overflow: hidden;
                white-space: nowrap;
                margin: 0;
                letter-spacing: 3px;
                animation: animatedline 3s steps(32) infinite;
                font-size: 22px; /* Increased font size */
                color: #ffa600; /* Text color */
                font-family: 'Montserrat', sans-serif; /* Change to a different font */
            }
            @keyframes animatedline {
                0% {
                    width: 0;
                }
                100% {
                    width: 100%;
                }
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="animated-line">Made by Nitin Pal & Dr. Girish Gidaye.</p>', unsafe_allow_html=True)

st.set_page_config(
    page_title="Stroke Predictor App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


color = [
    "linear-gradient(to right, #aed9f2, #6fa8dc, #428ecf, #2a6db2, #1d4f8a, #163d67, #102c44, #081b22, #030b0e)",
    "linear-gradient(to right, #c7f2ca, #8fd19a, #5db56e, #419e5b, #308148, #246d39, #185a2b, #0e461e, #062e10)",
    "linear-gradient(to right, #ffc7bd, #ff9b8a, #ff7261, #ff4a43, #ff292e, #e02025, #c51b1f, #9c161a, #7a1114)",
    "linear-gradient(to right, #d4b1e2, #aa8dd2, #875eb9, #663f9f, #542d87, #421b6f, #31135c, #1f0b49, #100336)",
    "linear-gradient(to right, #f0b6d6, #df92c1, #cf6aad, #c04b9d, #b32e8e, #9c1c7b, #84116a, #6b0959, #520148)",
    "linear-gradient(to right, #FFC9A9, #FFBA91, #FF8F4E, #FF721F, #EE5700, #C74900, #B24100, #903B0A, #762B00)"
]

# Function to read image as base64
def get_img_as_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Path to your background image file
img_path = "img/bg2.jpg"  

# Convert image to base64
img = get_img_as_base64(img_path)

# CSS style for background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://wallpapers.com/images/hd/dark-gradient-gw5zb5u4231un4q7.jpg");
    background-size: cover;  /* Cover the entire container */
    background-repeat: no-repeat;
    background-attachment: fixed;  /* Fixed positioning */
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    animation: moveSnow 6s linear infinite;
}}

@keyframes moveSnow {{
    0% {{ background-position: 0 0; }}
    100% {{ background-position: 100% 100%; }}
}}

[data-testid="stSidebarNav"] span {{
    color: white;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""

# Apply the CSS style with background images
st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to get predictions for Kivy app
def make_prediction_for_kivy(age, bmi, avg_glucose_level, gender, work_type, hypertension, heart_disease,
                              ever_married, residence_type, smoking_status):
    # Create a DataFrame with the input data using categorical mapping
    input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type,
                            avg_glucose_level, bmi, smoking_status]])

    # Validate age and work type
    age = validate_age_work_type(age, work_type)

    # Normalize the input data
    normalized_input_data = normalize_input_data(input_data, scaler)

    # Validate input
    validate_input(age, bmi, avg_glucose_level)
    
    # Make prediction
    prediction, confidence = predict_stroke_risk(normalized_input_data, loaded_model)
    

    # Apply threshold
    threshold = 0.1
    updated_res = (confidence > threshold).astype(int)

    # Return the prediction result and confidence
    return updated_res, confidence

@st.cache_data()
def assesBMI(BMI, AGE):
    BMI = BMI * 70
    if BMI > 45 and AGE > 75:
        inf = """
        Note: Information is unreliable.
        BMI > 45 and age > 75.
        """
    elif BMI <= 10:
        inf = "BMI too low"
    elif BMI < 18.5 and BMI > 10:
        inf = "Underweight"
    elif BMI >= 18.5 and BMI < 25:
        inf = "Normal Weight"
    elif BMI >= 25 and BMI < 30:
        inf = "Overweight"
    elif BMI >= 30 and BMI < 35:
        inf = "Moderate Obesity"
    elif BMI >= 35 and BMI < 40:
        inf = "Strong Obesity"
    elif BMI >= 40:
        inf = "Extreme Obesity"
    return inf

# Sidebar navigation with three tabs
tabs = st.sidebar.radio("Navigation", ["Home Page", "Predictor App"])

if tabs == "Home Page":
    st.title("Stroke Prediction Web App")
    st.write("")
    
    # Load the image file from your computer
    image_path = "img/brain.jpg"  # Replace 'path/to/your/image.jpg' with the actual path to your image file
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    # Convert the image bytes to base64 format
    image_base64 = base64.b64encode(image_bytes).decode()
    # Generate the HTML code with the local image
    html_code = f"""
    <div style="text-align:center;">
        <img src="data:image/jpeg;base64,{image_base64}" alt="Brain Stroke using Machine Learning" width="500">
        <p style="text-align:center;">AI Powered Brain Stroke Prediction</p>
    </div>
    """
    # Display the HTML code in Streamlit
    st.markdown(html_code, unsafe_allow_html=True)
    
    st.write("")
    st.write("<span style='color: #d6cfd3;'>This web app project is dedicated to predicting the likelihood of strokes using advanced machine learning techniques. While the predictions serve as valuable insights, it's important to remember that this model is not a substitute for professional medical advice. The project encourages users to prioritize their health and consider consulting a doctor for a comprehensive diagnosis. Behind the scenes, the model is trained on a diverse dataset, leveraging cutting-edge methodologies to ensure accurate predictions. By harnessing the power of data science, this project aims to raise awareness about stroke risk factors and empower individuals to make informed decisions about their health.</span>", unsafe_allow_html=True)
    st.write("")
    
    tooltips = {
        'Select a feature for description': "",
        'Gender': "Biological sex can influence stroke risk. For example, males may have a higher risk than females.",
        'Age': "Age is a crucial factor in stroke risk. The risk generally increases with age.",
        'Hypertension': "Hypertension (high blood pressure) is a significant risk factor for stroke.",
        'Heart Disease': "Heart disease can contribute to stroke risk. The two conditions often share common risk factors.",
        'Ever Married': "Marital status may have some correlation with lifestyle factors affecting stroke risk.",
        'Work Type': "Occupation and work-related stress can impact stroke risk.",
        'Residence Type': "Urban and rural living may have different lifestyle factors influencing stroke risk.",
        'Average Glucose Level': "High glucose levels can contribute to cardiovascular issues, potentially increasing stroke risk.",
        'BMI': "Body Mass Index (BMI) is a measure of body fat and can be associated with stroke risk.",
        'Smoking Status': "Smoking is a well-known risk factor for stroke."
    }
    
    # Feature analysis inputs
    feature_to_analyze = st.selectbox("Know more about how each feature is making impact in this prediction algorithm.", list(tooltips.keys()))
    a = tooltips[feature_to_analyze]
    if a == tooltips['Select a feature for description']:
        pass
    else:
        st.success(tooltips[feature_to_analyze])  # Display tooltip
    st.write("")

    animated_line()
    st.write("")
    about_text = (
        "This application is designed to assess the risk of stroke using machine learning algorithms. If a stroke is suspected, "
        "a doctor must always be consulted. This is a medical emergency."
    )
    tooltip = "This prediction is based on a machine learning model trained on various health-related features. It estimates the likelihood of an individual experiencing a stroke in the near future. The confidence level represents the model's certainty in its prediction."
    st.markdown(f'<div style="background-color: #415E90; padding: 10px; border-radius: 10px;">'
        f'<span title="{tooltip}" style="cursor: help; font-family: Arial, sans-serif; font-size: 16px;">&#9432;</span> {about_text}', unsafe_allow_html=True)
    # Add an image or any other content here

elif tabs == "Predictor App":
    st.title("Stroke Risk Assessment")
    
    # Dropdowns and sliders for input features
    gender_options = ['Male', 'Female']
    hypertension_options = ['No', 'Yes']
    heart_disease_options = ['No', 'Yes']
    ever_married_options = ['No', 'Yes']
    work_type_options = ['Child', 'Government', 'Never worked', 'Private', 'Self-employed']
    residence_type_options = ['Urban', 'Rural']
    smoking_status_options = ['Never Smoked', 'Formerly Smoked', 'Smokes', 'Unknown']

    # Input for a single patient
    st.sidebar.title("Patient Data")
    gender = map_categorical(st.sidebar.selectbox('Gender', gender_options), {'Male': 1, 'Female': 0})
    work_type = map_categorical(st.sidebar.selectbox('Work Type', work_type_options, key='work_type'),
                                {'Child': 0, 'Government': 1, 'Never worked': 2, 'Private': 3, 'Self-employed': 4})

    # Adjust age range based on work type
    if work_type == 0:
        age_range = (0, 16)
    else:
        age_range = (17, 100)

    age = st.sidebar.slider('Age', age_range[0], age_range[1], age_range[0], help='Select the age of the patient')  # Default age set to minimum value
    age = validate_age_work_type(age, work_type)

    hypertension = map_categorical(st.sidebar.selectbox('Hypertension', hypertension_options), {'No': 0, 'Yes': 1})
    heart_disease = map_categorical(st.sidebar.selectbox('Heart Disease', heart_disease_options), {'No': 0, 'Yes': 1})
    ever_married = map_categorical(st.sidebar.selectbox('Ever Married', ever_married_options), {'No': 0, 'Yes': 1})
    residence_type = map_categorical(st.sidebar.selectbox('Residence Type', residence_type_options, key='residence_type'),
                                     {'Urban': 1, 'Rural': 0})
    avg_glucose_level = st.sidebar.number_input('Average Glucose Level', 0.0, 400.0, 200.0, step=0.1) / 400
    bmi = st.sidebar.number_input('BMI', 0.0, 70.0, 25.0, step=0.1) / 70
    smoking_status = map_categorical(st.sidebar.selectbox('Smoking Status', smoking_status_options, key='smoking_status'),
                                     {'Never Smoked': 0, 'Formerly Smoked': 1, 'Smokes': 2, 'Unknown': 3})

    # Create a DataFrame with the input data using categorical mapping
    input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type,
                            avg_glucose_level, bmi, smoking_status]])

    # Validate age and work type
    age = validate_age_work_type(age, work_type)

    # Normalize the input data
    normalized_input_data = normalize_input_data(input_data, scaler)
    
        

    # Button to trigger prediction
    if  st.button('Predict 🚀', key='predict_button', help='Click to predict stroke risk'):
        # Validate input
        validate_input(age, bmi, avg_glucose_level)
        # # Display a brief animation for prediction
        # with st.spinner(text='Predicting...'):
        #     time.sleep(1)  # Simulate a 1-second delay for the animation
        # Make prediction
        prediction, confidence = predict_stroke_risk(normalized_input_data, loaded_model)

        
        # Display a customized progress bar
        st.markdown(
            f"""
            <style>
                .custom-progress-bar-container {{
                    width: 100%;
                    background-color: #ffffff;
                    border-radius: 5px;
                    overflow: hidden;
                }}
    
                .custom-progress-bar {{
                    height: 25px;
                    width: 0%;
                    
                    background: {random.choice(color)};
                    border-radius: 5px;
                    color: white;
                    text-align: center;
                    line-height: 24px;
                    transition: width 0.5s ease-in-out;
                }}
            </style>
            """
            , unsafe_allow_html=True
        )
    
        progress_bar_container = st.empty()
    
        # Simulate a delay (replace this with your actual prediction process)
        for percent_complete in range(0, 101, 10):
            time.sleep(0.23)
            progress_bar_container.markdown(
                f"""
                <div class="custom-progress-bar-container">
                    <div class="custom-progress-bar" style="width: {percent_complete}%;">
                        Predicting... {percent_complete}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        # Clear the progress bar
        progress_bar_container.empty()

    
        # Apply threshold
        threshold = 0.1
        updated_res = (confidence > threshold).astype(int)

        # Output the result
        # st.write(f"Prediction: {prediction[0]}")
        # Apply threshold and print additional information
        thresholded_class = 1 if confidence > threshold else 0
        # st.write(f"Thresholded Class: {thresholded_class}")

        if updated_res == 0:
            st.success("Decision: The person is Healthy")
            st.info("The likelihood of experiencing a stroke in the near future is relatively low. Nevertheless, it is crucial to maintain good self-care habits.")
        else:
            st.error("Decision: The person may be at risk of a Stroke")
            st.warning("There is a possibility that you might experience a stroke soon or are currently at risk. Consequently, it is strongly recommended to prioritize and enhance your self-care practices.")

        # Before applying the threshold
        # st.write(f"Raw Predictions: {prediction}")
        # st.write(f"Confidence Values: {confidence}")

        # Create a tooltip-like effect on hover
        # tooltip1 = """This is the indication for the risk of stroke, as in how much confident the model is about this prediction."""
        # st.markdown(f'<span title="{tooltip1}" style="cursor: help; font-family: Arial, sans-serif; font-size: 16px;">Confidence Level &#9432;</span>', unsafe_allow_html=True)
        # confidence_level = round((1-confidence[0]) * 100, 2)
        # if confidence_level <= 80:
        #     st.warning(f"Confidence level in the risk assessment: {confidence_level}%")
        # if confidence_level > 80:
        #     st.info(f"Confidence level in the risk assessment: {confidence_level}%")
            
        tooltip1 = """Indicating your weight description based on BMI."""
        st.markdown(f'<span title="{tooltip1}" style="cursor: help; font-family: Arial, sans-serif; font-size: 16px;">BMI Level: &#9432;</span>', unsafe_allow_html=True)
        st.info(assesBMI(bmi, age))

