import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import json

with open("columns.json", "r") as f:
    __data_columns = json.load(f)['data_columns']
    __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

# __locations = None
# __data_columns = None
# __model = None

st.write(
    """
    # House Price Prediction
    This app predict price of **Bangalore** 

    """
)
st.sidebar.header('User Input Features')


def user_input_features():
    square_fit = st.sidebar.slider('Sqaure Fit', 500, 5000, 800)
    bhk = st.sidebar.selectbox('BHK', (1, 2, 3, 4, 5))
    bath = st.sidebar.selectbox('BATH', (1, 2, 3, 4, 5,))
    location = st.sidebar.selectbox('Location', (__locations))

    data = {
        'location': location,
        'square_fit': square_fit,
        'bath': bath,
        'bhk': bhk
    }
    return location, square_fit, bath, bhk


df = user_input_features()
# st.subheader('Class labels and their corresponding index number')
st.write('Location - ', df[0])
st.write('Sqaure_fit - ', df[1])
st.write('BHK - ', df[2])
st.write('Bath - ', df[3])

# Reads in saved Regression Model
__model = pickle.load(open('bhp.pkl', 'rb'))


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


# Apply model to make predictions
prediction = get_estimated_price(df[0], df[1], df[2], df[3])

st.subheader('Prediction')
st.write(prediction, 'Lakhs')
