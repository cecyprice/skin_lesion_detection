import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.misc import imread
# from skin_lesion_detection.encoders import ImageScaler
import joblib
from skin_lesion_detection.predict import Preprocessor

from skin_lesion_detection import encoders as encoders
from skin_lesion_detection import trainer


#

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

# create df that mimics trainer dataframe
df = pd.DataFrame({
          'sex': str(sex).lower(),
          'age': float(age),
          'localization': str(les_loc).lower(),
          'dx_type': str(tech_val_field).lower(),
          }, index=[0])

reshaped = image.reshape(810000)
resized_reshaped = np.resize(image, (75, 100, 3)).reshape(22500)

image_list = [i for i in reshaped]
image_resized_list = [i for i in resized_reshaped]

df['images'] = [image_list]
df['images_resized'] = [image_resized_list]



st.dataframe(df)

# apply pipeline transformations to df


# preprocessor = Preprocessor().predict(df)
model = joblib.load(open("skin_lesion_detection/pipeline_1.joblib", 'rb'))

# preproc = Preprocessor.from_path()
# df_preproc = preproc.transform(df)

# st.write(df_preproc)
name = 'vgg_test'
json_file = open(f'{name}', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
model = loaded_model.load_weights(f'{name}.h5')

model.predict(df)



# split into X_met and X_im

# predict on gcp saved model (or local saved model)
