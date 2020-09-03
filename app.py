import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.misc import imread
from keras.models import load_model
import joblib

import tensorflow.keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, concatenate, Dropout, Activation, MaxPooling2D, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16


# disable warning
st.set_option('deprecation.showfileUploaderEncoding', False)

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
          'NA': float(0),
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

st.dataframe(df)

# resize image and scale using ImageScaler
image = image
resized_image = np.resize(image, (75, 100, 3))

# split into X_met and X_im
X_met_test = df.astype('float64')
X_im_test = resized_image # toggle between image and resized_image


# import model and make prediction
st.markdown("""## Step 3: Get prediction""")
prediction = st.button("Predict")


# build model
mlf_fork = Sequential()
mlf_fork.add(Dense(16, input_dim=23, activation="relu"))
mlf_fork.add(Dense(8, activation="relu"))
mlf_fork.add(Dense(4, activation="relu"))

cnn_fork = VGG16(weights='imagenet', input_shape=(75, 100, 3), include_top=False)
for layer in cnn_fork.layers:
  layer.trainable = False
inp = Input(shape=(75, 100, 3))
base_output = cnn_fork(inp)
x = Flatten()(base_output)
x = Dense(126, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(4, activation='relu')(x)
cnn_fork = Model(inp, x)

combinedInput = concatenate([mlf_fork.output, cnn_fork.output])
x = Dense(4, activation="relu")(combinedInput)
x = Dense(7, activation="softmax")(x)
model = Model(inputs=[mlf_fork.input, cnn_fork.input], outputs=x)

# load weights and make prediction
X_im_test = X_im_test.reshape(1, 75, 100, 3)


translate_dict = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions ',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'}

if prediction:
  # model = load_model('skin_lesion_detection/tl_vgg_1.h5')
  model.load_weights('skin_lesion_detection/tl_vgg_1.h5')
  results = model.predict(x=[X_met_test, X_im_test])

  indices = np.argsort(results).tolist()[0]

  dict_nums = {}
  for i, val in enumerate(indices):
    dict_nums[i] = translate_dict[val]


  st.markdown(f"""### Top 3 most likely diagnoses:)
  #### 1) {dict_nums[6]}
  #### 2) {dict_nums[5]}
  #### 3) {dict_nums[4]}""")
  ##### accuracy = {}""")

  # diplay top 3 most likely predictions with accuracy


