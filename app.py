import streamlit as st
import pandas as pd
from tensorflow import keras
from geopy.geocoders import Nominatim 
import haversine as hs
import numpy as np
import pickle




st.write("""
# Credit Card Transaction #
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    city = st.sidebar.text_input('Card Holder City', "Boston")
    state = st.sidebar.text_input('Card Holder State', "MA")
    amount = float(st.sidebar.text_input('Amount', 0))
    age = float(st.sidebar.text_input('Age', 0))
    gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
    city_M = st.sidebar.text_input('Merchant City', "New York")
    state_M = st.sidebar.text_input('Merchant State', "NY")
    category = st.sidebar.selectbox('Category', ('Gas Transport', 'Grocery Pos', 'Home', 'Shopping Pos',
     'Kids Pets', 'Shopping Net', 'Entertainment', 'Restaurant', 'Health Care', 'Fitness', 'Misc Pos', 'Misc Net', 'Grocery Net', 'Travel'))
   
    geolocator = Nominatim(user_agent='myapplication')
    location = geolocator.geocode(city + ' ' + state)
    location_M = geolocator.geocode(city_M + ' ' + state_M)
    lat = location.latitude
    lon = location.longitude
    lat_M = location_M.latitude
    lon_M = location_M.longitude
    distance = hs.haversine((lat, lon),(lat_M, lon_M))
    

    data = {'distances': distance,
            'amt': amount,
            'age': age,
            'gender_M': gender,
            #'merchant city': city_M,
            #'merchant state': state_M,
            'category': category}
    features = pd.DataFrame(data, index=[0])
    cats = ['Gas Transport', 'Grocery Pos', 'Home', 'Shopping Pos',
     'Kids Pets', 'Shopping Net', 'Entertainment', 'Restaurant', 'Health Care', 'Fitness',
     'Misc Pos', 'Misc Net', 'Grocery Net', 'Travel']

    category_dict = {'Restaurant': 'food_dining','Gas Transport': 'gas_transport', 'Grocery Net': 'grocery_net',
    'Grocery Pos': 'grocery_pos', 'Fitness':'health_fitness','Home': 'home', 'Kids Pets':'kids_pets',
    'Misc Net': 'misc_net', 'Misc Pos':'misc_pos', 
    'Health Care': 'personal_care', 'Entertainment': 'entertainment',
    'Shopping Net': 'shopping_net', 'Shopping Pos': 'shopping_pos', 'Travel': 'travel'}

    cats = ['category_' + category_dict[cat] for cat in cats]

    cat = pd.DataFrame(columns=cats, data=np.zeros((1, len(cats))), index=[0])

    cat['category_'+ category_dict[category]] = 1.

    df = pd.concat([features, cat], axis=1)

    df['gender_M'] = df['gender_M'].map({'Female': 0, 'Male':1})
    df = df.drop(columns = ['category'])

    df = df[['amt', 'distances', 'category_food_dining', 'category_gas_transport',
       'category_grocery_net', 'category_grocery_pos',
       'category_health_fitness', 'category_home', 'category_kids_pets',
       'category_misc_net', 'category_misc_pos', 'category_personal_care',
       'category_shopping_net', 'category_shopping_pos', 'category_travel',
       'gender_M', 'age']]

    return df

df = user_input_features()

#st.subheader('User Input parameters')
#st.write(df)

scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))
X_sc = scaler.transform(df)
model = keras.models.load_model('fraud_detection.h5')

# encode = ['category','gender']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col) 
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
# df = df[:1] # Selects only the first row (the user input data)




# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target

# clf = RandomForestClassifier()
# clf.fit(X, Y)

prediction = model.predict(X_sc)

st.subheader('Prediction')
if prediction == 0:
    st.write("It's not a fraudulent transaction")
else:
    st.write("It's a Fradulent transaction")

import matplotlib.pyplot as plt

uploaded_file = st.file_uploader('Choose a file')

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    gr = df.groupby('category')['amt'].sum() / df['amt'].sum() * 100

    fig, ax= plt.subplots()
    ax.pie(gr, autopct='%.1f%%', labels=gr.index)
    ax.axis('equal')

    st.pyplot(fig)
else:
    st.warning("you need to upload a csv file.")


# prediction_proba = clf.predict_proba(df)

# st.subheader('Class labels and their corresponding index number')
# st.write(iris.target_names)

# st.subheader('Prediction')
# st.write(iris.target_names[prediction])
# #st.write(prediction)

# st.subheader('Prediction Probability')
# st.write(prediction_proba)