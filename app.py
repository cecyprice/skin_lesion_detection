import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.misc import imread
# from skin_lesion_detection.encoders import ImageScaler
import joblib
from skin_lesion_detection.predict import Preprocessor


st.markdown("""# Skin Lesion Detection Engine""")

# Step 1: personal details
st.markdown("""## Step 1: Enter personal details""")

sex = st.selectbox("👇 Select Sex", ["Male", "Female"])
age = st.text_input("👇 Enter Age", "-")
les_loc = st.selectbox("👇 Select Lesion Location", ["Trunk", "Back", "Abdomen", "Upper extremity", "Lower extremity", "Foot", "Chest", "Face", "Neck", "Genitals", "Hand", "Scalp", "Ear", "Acral", "Other/NA"])
tech_val_field = st.selectbox("👇 Select Technical Validation Field", ["Histopathology", "Confocal", "Follow-up", "Consensus"])


# Step 2: photo upload
st.markdown("""## Step 2: Upload photograph
### Upload .jpeg image of your lesion """)
uploaded_image = st.file_uploader("Select file")

if uploaded_image is not None:
    # Storing the image into a NumPy array and plotting it
    image = imread(uploaded_image)
    st.image(image, use_column_width = True)

# reassign to match data categories
if str(les_loc).lower() == "genitals":
    les_loc = "genital"
if str(les_loc).lower() == "other/na":
    les_loc = "unknown"
if str(tech_val_field).lower() == "histopathology":
    tech_val_field = "histo"

# create df that mimics trainer dataframe wtih transformations
df = pd.DataFrame({
          'acral': float(0),
          'abdomen' : float(0),
          'back' : float(0),
          'chest' : float(0),
          'ear' : float(0),
          'face' : float(0),
          'foot' : float(0),
          'genital' : float(0),
          'hand' : float(0),
          'lower extremity': float(0),
          'neck' : float(0),
          'scalp' : float(0),
          'trunk' : float(0),
          'upper extremity' : float(0),
          'unknown' : float(0),
          'confocal': float(0),
          'consensus': float(0),
          'follow_up': float(0),
          'histo': float(0),
          'female': float(0),
          'male': float(0),
          'unknown': float(0),
          'age_scaled': (float(age)-60)/25
          }, index=[0])


# enter localization info value into dataframe
loc_list = ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
       'genital', 'hand', 'lower extremity', 'neck', 'scalp', 'trunk',
       'unknown', 'upper extremity']
for i in loc_list:
    if str(les_loc).lower() == i:
        df.set_value(0, i, float(1))

# enter dx_type info value into dataframe
dx_list = ['histo', 'confocal', 'consensus', 'follow_up']
for i in dx_list:
    if str(tech_val_field).lower() == i:
        df.set_value(0, i, float(1))

# enter sex info value into dataframe
sex_list = ['male', 'female']
for i in sex_list:
    if str(sex).lower() == i:
        df.set_value(0, i, float(1))

# st.dataframe(df) to see df

# resize image and scale using ImageScaler
image = image
resized_image = np.resize(image, (75, 100, 3))

# split into X_met and X_im
X_met_test = df.astype('float64')
X_im_test = resized_image # toggle between image and resized_image



st.write(X_met_test)
st.write(X_im_test)


# predict on gcp saved model (or local saved model)
