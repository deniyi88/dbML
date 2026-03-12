
    # Diabetes Prediction Web App using Streamlit

    # This application loads a pre-trained Random Forest model to predict the likelihood 
    # of Type 2 Diabetes based on user-inputted health indicators. It also displays 
    # the top contributing factors for the prediction using feature importance scores.

    # Features:
    # - Custom landing page with styled background
    # - Styled "Start Prediction" button for better user engagement
    # - Interactive input form for health metrics
    # - Real-time prediction and result display with top feature insights
    # - Modular layout with clean navigation between pages


import pickle  # For loading the saved model
import streamlit as st  # Streamlit library for building the web app
import numpy as np  # For numerical operations
import base64  # For encoding background image

# ----------------------------
# Load the trained model from a pickle file
# ----------------------------
with open(r'C:/Users/dell/OneDrive - Teesside University/Dissertation/Submission/D3310898-Samson-Adeyemi-Artefact/rf_model.pkl', 'rb') as pickle_in:
    model = pickle.load(pickle_in)

# ----------------------------
# Extract feature importance directly from the loaded model
# ----------------------------
feat_imp = model.feature_importances_

# Define the names of the features in the same order as used in training
feature_names = ["Age", "Pulse Rate", "Systolic BP", "Diastolic BP", "Glucose",
                 "Weight", "BMI", "Family Diabetes", "Hypertensive"]

# ----------------------------
# Function to set a background image for the Streamlit app
# ----------------------------
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Function to calculate the top contributing features
# ----------------------------
def get_top_features(input_values, feat_imp, feature_names, top_n=3):
    # Multiply each input value with its corresponding feature importance
    importance_scores = [(feature_names[i], feat_imp[i] * abs(input_values[i])) for i in range(len(input_values))]
    # Sort the features by their contribution (highest to lowest)
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    return importance_scores[:top_n]  # Return top N contributing features

# ----------------------------
# Landing page with welcome message and a start button
# ----------------------------
def show_landing_page():
    set_background('homepage.webp')  # Set background for landing page only

    st.title("Welcome to the Diabetes Prediction App 👋")

    st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.85); padding: 20px; border-radius: 10px;">
            <h3>This tool helps predict the likelihood of Type 2 Diabetes based on health indicators.</h3>
            <p>Click the button below to begin the prediction process.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # When user clicks "Start Prediction", navigate to prediction page
    if st.button("**Start Prediction**"):
        st.session_state.page = 'predict'
        st.rerun()

# ----------------------------
# Prediction page where users input their health data
# ----------------------------
def show_prediction_page():
    # Styled title banner
    html_temp = """
    <div style="background-color:#b9f2ff; padding:13px">
    <h1 style="color:black; text-align:center;">Diabetes Prediction</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create two columns for cleaner input layout
    col1, col2 = st.columns([1, 1])

    # User inputs in left column
    with col1:
        age = st.number_input('Age (8 to 112)', min_value=8, max_value=112)
        pulse_rate = st.number_input('Pulse Rate (5 to 133)', min_value=5, max_value=133)
        systolic_bp = st.number_input('Systolic Blood Pressure (62 to 231)', min_value=62, max_value=231)
        diastolic_bp = st.number_input('Diastolic Blood Pressure (45 to 119)', min_value=45, max_value=119)
        glucose = st.number_input('Glucose (0 to 34)', min_value=0, max_value=34)

    # User inputs in right column
    with col2:
        weight = st.number_input('Weight (3 to 101)', min_value=3, max_value=101)
        bmi = st.number_input('BMI (1 to 156)', min_value=1, max_value=156)
        family_diabetes = st.selectbox('Family History of Diabetes', ['Yes', 'No'])
        hypertensive = st.selectbox('Hypertensive', ['Yes', 'No'])

    # When the Predict button is clicked
    if st.button('**Predict**'):
        try:
            # Prepare input data as a NumPy array
            input_data = np.array([[age, pulse_rate, systolic_bp, diastolic_bp, glucose,
                                    weight, bmi,
                                    1 if family_diabetes == 'Yes' else 0,
                                    1 if hypertensive == 'Yes' else 0]])
            
            # Make prediction using the model
            prediction = model.predict(input_data)

            # Translate prediction result to human-readable format
            result = {0: 'No Diabetes', 1: 'Diabetes Detected'}
            pred_text = result.get(prediction[0], "Unknown")

            # Conditional styling for result display
            if prediction[0] == 1:
                color = "#FFCCCC"  # Light red for positive case
                text_color = "red"
            else:
                color = "#CCFFCC"  # Light green for negative case
                text_color = "green"

            # Display prediction result in a styled box
            st.markdown(f"""
                <div style="background-color: {color}; padding: 15px; border-radius: 10px;">
                    <h3 style="color: {text_color}; text-align: center;">Result: {pred_text}</h3>
                </div>
            """, unsafe_allow_html=True)

            # Display top contributing features based on input and model importance
            top_features = get_top_features(input_data[0], feat_imp, feature_names)
            st.subheader("Top Contributing Factors:")
            for feature, importance in top_features:
                st.write(f"- **{feature}** contributed with importance score: {importance:.4f}")

        except ValueError as e:
            # Handle any conversion or input errors
            st.error(f"Error: {str(e)}. Please enter valid values.")

    # Button to go back to landing page
    if st.button('**Back to Home**'):
        st.session_state.page = 'home'
        st.rerun()

# ----------------------------
# Main control function to navigate between pages
# ----------------------------
def main():
    # Initialize session state if not already set
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Route to the appropriate page
    if st.session_state.page == 'home':
        show_landing_page()
    elif st.session_state.page == 'predict':
        show_prediction_page()

# Entry point for the Streamlit app
if __name__ == '__main__':
    main()
