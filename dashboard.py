import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

# Load the trained model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Load the feature names used during training
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)


# List of columns used during training (use these columns for both training and prediction)
TRAINING_COLUMNS = [
    'Fuel Type', 'Body Type', 'Transmission Type', 'Manufacturer', 'Model', 
    'Variant Name', 'Location', 'RTO', 'Transmission', 'Engine Type', 
    'Value Configuration', 'Fuel Supply System', 'Turbo Charger', 
    'Super Charger', 'Drive Type', 'Steering Type', 'Front Brake Type', 
    'Rear Brake Type', 'Tyre Type', 'Number of Owners', 'Insurance Validity'
]
# Function to preprocess data
def preprocess_data(df):
    # Handle missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[df.select_dtypes(include=[np.number]).columns] = num_imputer.fit_transform(df.select_dtypes(include=[np.number]))
    df[df.select_dtypes(exclude=[np.number]).columns] = cat_imputer.fit_transform(df.select_dtypes(exclude=[np.number]))

    # Check if columns in TRAINING_COLUMNS exist in input data
    nominal_cols = [col for col in TRAINING_COLUMNS if col in df.columns and df[col].dtype == 'object']
    
    # One-hot encoding for nominal columns
    df = pd.get_dummies(df, columns=nominal_cols)

    # Label encoding for ordinal columns
    ordinal_cols = ['Number of Owners', 'Insurance Validity']
    le = LabelEncoder()
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Normalize numerical columns
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Align the columns of input data with those in training
    df = df.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    
    return df



# Streamlit app
st.title("Car Price Prediction App")

# Define a form for user input
def user_input_features():
    Fuel_Type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG'])
    Body_Type = st.sidebar.selectbox('Body Type', ['SUV', 'Sedan', 'Hatchback', 'MUV', 'Minivans'])
    Kilometers_Driven = st.sidebar.number_input('Kilometers Driven', min_value=0, max_value=500000, value=120000)
    Transmission_Type = st.sidebar.selectbox('Transmission Type', ['Manual', 'Automatic'])
    Number_of_Owners = st.sidebar.selectbox('Number of Owners', [1, 2, 3, 4, 5])
    Manufacturer = st.sidebar.selectbox('Manufacturer', ['Ford', 'Hyundai', 'Maruti', 'Honda', 'Tata', 'Kia', 'Renault',
                                                        'Volkswagen', 'Mahindra', 'Fiat', 'MG', 'Skoda', 'BMW', 'Datsun',
                                                        'Toyota', 'Chevrolet', 'Citroen', 'Nissan', 'Hindustan Motors'])
    Model = st.sidebar.selectbox('Model', ['Ford Ecosport', 'Hyundai Xcent', 'Hyundai Venue', 'Maruti Baleno',
                                            'Hyundai Grand i10', 'Honda Jazz', 'Hyundai i20', 'Tata Nexon',
                                            'Maruti Swift', 'Hyundai Santro', 'Maruti Ertiga',
                                            'Maruti Alto 800', 'Kia Seltos', 'Renault KWID', 'Maruti Celerio',
                                            'Renault Lodgy', 'Volkswagen Polo', 'Mahindra KUV 100',
                                            'Mahindra XUV300', 'Honda Brio', 'Renault Kiger', 'Hyundai Creta',
                                            'Maruti Alto K10', 'Maruti Ignis', 'Toyota Innova',
                                            'Renault Captur', 'Maruti Vitara Brezza', 'Maruti Swift Dzire',
                                            'Fiat Linea', 'Maruti SX4 S Cross', 'Maruti Wagon R',
                                            'Volkswagen Taigun', 'MG Astor', 'Skoda Rapid', 'BMW X3',
                                            'Hyundai Verna', 'Kia Sonet', 'Maruti S-Presso', 'Maruti Ritz',
                                            'Datsun RediGO', 'Ford Aspire', 'Ford Freestyle', 'Ford Figo',
                                            'Tata Tigor', 'Hyundai i10', 'Toyota Glanza', 'Tata Tiago',
                                            'Maruti Celerio X', 'Toyota Urban cruiser', 'Hyundai i20 Active',
                                            'Hyundai Elantra', 'Tata Altroz', 'Tata Indica V2',
                                            'Volkswagen Ameo', 'Honda City', 'Mahindra TUV 300',
                                            'Chevrolet Beat', 'Maruti Eeco', 'Hyundai Grand i10 Nios',
                                            'Mahindra Quanto', 'Hyundai i20 N Line', 'Citroen C3',
                                            'Tata Manza', 'Fiat Punto', 'Chevrolet Enjoy', 'Tata Sumo',
                                            'Tata Indigo', 'Toyota Etios', 'Honda WR-V', 'Renault Duster',
                                            'Fiat Punto EVO', 'Hyundai Alcazar', 'Renault Fluence',
                                            'Chevrolet Sail', 'Datsun GO', 'Maruti XL6', 'Maruti Ertiga Tour',
                                            'MG Hector', 'Hyundai Aura', 'Honda Amaze', 'Kia Carens',
                                            'Nissan Micra', 'Tata Punch', 'Honda BR-V', 'Tata Tiago NRG',
                                            'Nissan Magnite', 'Fiat Grande Punto', 'Mahindra KUV 100 NXT',
                                            'Chevrolet Aveo', 'Tata Indica', 'Fiat Punto Pure',
                                            'Chevrolet Optra', 'Nissan Kicks', 'Maruti Grand Vitara',
                                            'Toyota Etios Cross', 'Tata Zest', 'Maruti Swift Dzire Tour',
                                            'Honda Mobilio', 'Ford Fiesta Classic', 'Maruti Brezza',
                                            'Ford Fiesta', 'Fiat Avventura', 'Hyundai Xcent Prime',
                                            'Mahindra Scorpio', 'Mahindra Bolero Neo', 'Honda Civic',
                                            'Toyota Etios Liva', 'Renault Pulse', 'Maruti Jimny',
                                            'Chevrolet Spark', 'Ambassador'])
    Model_Year = st.sidebar.number_input('Model Year', min_value=2000, max_value=2024, value=2015)
    Variant_Name = st.sidebar.text_input('Variant Name', 'VXI')
    Location = st.sidebar.selectbox('Location', ['bangalore', 'chennai', 'hyderabad', 'jaipur', 'kolkata'])
    Seats = st.sidebar.slider('Seats', 2, 9, 5)
    Engine_Displacement = st.sidebar.number_input('Engine Displacement', min_value=500, max_value=5000, value=998)
    Mileage = st.sidebar.number_input('Mileage', min_value=5.0, max_value=30.0, value=23.1)
    Max_Power = st.sidebar.number_input('Max Power', min_value=50.0, max_value=500.0, value=67.04)
    Torque = st.sidebar.number_input('Torque', min_value=50.0, max_value=500.0, value=90.0)
    Color = st.sidebar.selectbox('Color', ['White', 'Others', 'Red', 'Orange', 'Grey', 'Blue', 'Silver',
                                            'Yellow', 'Brown', 'Maroon', 'Gray', 'Black', 'Green', 'Gold',
                                            'MODERN STEEL METALLIC', 'Golden', 'Star Dust', 'Flash Red',
                                            'Wine Red', 'T Wine', 'Prime Star Gaze', 'Golden brown',
                                            'SILKY SILVER', 'BERRY RED', 'PREMIUM AMBER METALLIC', 'COPPER',
                                            'CARNELIAN RED PEARL', 'Purple', 'POLAR WHITE', 'Hip Hop Black',
                                            'Nexa Blue', 'Silky Silver', 'Polar White', 'magma gray', 'CBeige',
                                            'm grey', 'Granite Grey', 'Phantom Black', 'Metallic Magma Grey',
                                            'Metallic Glistening Grey', 'Fiery Red', 'Superior white',
                                            'Sleek Silver', 'Smoke Grey', 'Pearl Arctic White', 'Silky silver',
                                            'Gravity Gray', 'Metallic Premium silver', 'Glistening Grey',
                                            'PLATINUM WHITE PEARL', 'Pearl Met. Arctic White',
                                            'Metallic silky silver', 'Pure white', 'Diamond White', 'Ray blue',
                                            'Candy White', 'Daytona Grey', 'Moonlight Silver',
                                            'Aurora Black Pearl', 'StarDust', 'Glacier White Pearl',
                                            'Cashmere', 'Foliage', 'Bronze', 'Outback Bronze', 'Cherry Red',
                                            'Sunset Red', 'Light Silver', 'Dark Blue'])

    # Create a DataFrame
    data = {
        'Fuel Type': Fuel_Type,
        'Body Type': Body_Type,
        'Kilometers Driven': Kilometers_Driven,
        'Transmission Type': Transmission_Type,
        'Number of Owners': Number_of_Owners,
        'Manufacturer': Manufacturer,
        'Model': Model,
        'Model Year': Model_Year,
        'Variant Name': Variant_Name,
        'Location': Location,
        'Seats': Seats,
        'Engine Displacement': Engine_Displacement,
        'Mileage': Mileage,
        'Max Power': Max_Power,
        'Torque': Torque,
        'Color': Color
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Preprocess the input data
input_df_processed = preprocess_data(input_df)

# Align input data with the feature names used during training
input_df_processed = input_df_processed.reindex(columns=feature_names, fill_value=0)

# Predict using the loaded model
prediction = model.predict(input_df_processed)

st.write(f"## Predicted Price: ₹{prediction[0]:,.2f}")

# Preprocess the input data
input_df_processed = preprocess_data(input_df)

