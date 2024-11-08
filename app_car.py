
import streamlit as st
import pandas as pd
import joblib

# --- Load the model and encoders (for later use) ---
loaded_model = joblib.load('C:/Users/SABA ANBU/OneDrive/Desktop/project3/rf_trainedmodel.pkl')
loaded_encoders = joblib.load('C:/Users/SABA ANBU/OneDrive/Desktop/project3/label_encoders.pkl')
loaded_scaler = joblib.load('C:/Users/SABA ANBU/OneDrive/Desktop/project3/min.pkl')
ridge_model = joblib.load('C:/Users/SABA ANBU/OneDrive/Desktop/project3/ridge_model.pkl')


categorical_columns = [
    "Body_type",
    "Transmission",
    "Original_equipment_manufacturer",
    "Model",
    "Variant_name",
    "Insurance_validity",
    "Fuel_type",
    "Colour",
    "Location",
]


# --- Function to take custom inputs and predict price ---
@st.cache_resource
def predict_price(
    mileage,
    engine_displacement,
    year_of_manufacture,
    transmission,
    fuel_type,
    owner_no,
    model_year,
    location,
    kilometer_driven,
    body_type,
):

    # Create a DataFrame from the input values
    input_data = pd.DataFrame(
        {
            "Mileage": [mileage],
            "Engine_displacement": [engine_displacement],
            "Year_of_manufacture": [year_of_manufacture],
            "Transmission": [transmission],
            "Fuel_type": [fuel_type],
            "Owner_No.": [owner_no],
            "Model_year": [model_year],
            "Location": [location],
            "Kilometer_Driven": [kilometer_driven],
            "Body_type": [body_type],
        }
    )

    # Encode categorical features using the loaded LabelEncoders
    for col in categorical_columns:
        if col not in {
            "Original_equipment_manufacturer",
            "Model",
            "Variant_name",
            "Insurance_validity",
            "Colour",
        }:
            le = loaded_encoders[col]
            input_data[col] = le.transform(input_data[col].astype(str))

    # Make the prediction
    predicted_price = loaded_model.predict(input_data)
    predicted_price_norm = loaded_scaler.inverse_transform([[predicted_price[0]]])[0][0]
    return predicted_price_norm


# Streamlit UI
st.set_page_config(page_title="Car Dhekho - Used Car Price Prediction", layout="wide")
st.title(":red_car: Car Dhekho - Used Car Price Prediction")
st.markdown(
    "<style>div.block-container{padding-top:2rem;}</style", unsafe_allow_html=True
)

# URL of the image
image_url = 'https://www.carz4sale.in/pictures/static/used-car-valuation.png'

# Display the image
st.image(image_url, use_column_width=True)


st.sidebar.header("Select Features")

# Input fields
mileage = st.sidebar.slider("Mileage", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
engine_displacement = st.sidebar.slider("Engine Displacement (cc)", min_value=740, max_value=2000, value=1500, step=10)
year_of_manufacture = st.sidebar.slider( "Year of Manufacture", min_value=2000, max_value=2024, value=2015)

transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel"])
owner_no = st.sidebar.number_input("Owner No.", min_value=0)
model_year = st.sidebar.number_input("Model Year", min_value=2000, max_value=2024)
# Dropdown for Location
location = st.sidebar.selectbox("Location",["Chennai", "Bangalore", "Delhi", "Kolkata", "Jaipur", "Hyderabad"])

# Dropdown for Body Type
body_type = st.sidebar.selectbox( "Body Type",["Hatchback", "SUV", "Sedan", "MUV", "Minivans", "Coupe", "Pickup Trucks", "Convertibles", "Hybrids", "Wagon"])

#kilometer_driven = st.sidebar.number_input("Kilometer Driven", min_value=0)
kilometer_driven  = st.sidebar.slider( "Kilometer Driven", min_value=0, max_value=5500000, value=2015)

if st.sidebar.button("Estimate Used Car Price"):
    try:
        predicted_price = predict_price(
            mileage,
            engine_displacement,
            year_of_manufacture,
            transmission,
            fuel_type,
            owner_no,
            model_year,
            location,
            kilometer_driven,
            body_type,
        )
        st.write(f"Predicted Price: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
