import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_wellness_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a wellness tourism package based on their profile.
Enter the customer details below to get a prediction.
""")

# Input fields tailored to the Wellness Tourism dataset features
age = st.number_input("Age", 18, 100, 35)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 0, 120, 15)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
numberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
number_of_trips = st.number_input("Number of Trips per Year", 0, 20, 2)
passport = st.selectbox("Has Passport", [0, 1])
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Owns Car", [0, 1])
number_of_children_visiting = st.number_input("Number of Children Visiting (under 5)", 0, 5, 0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", 0.0, 200000.0, 20000.0, 500.0)

# Build input dataframe matching training columns
input_data = pd.DataFrame([{
    "Age": age,
    "Type of Contact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": numberOfFollowups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Package Taken (ProdTaken: Yes)" if prediction == 1 else "Package Not Taken (ProdTaken: No)"
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The model predicts: **{result}**")
    else:
        st.info(f"The model predicts: **{result}**")
