import os
import numpy as np
import traceback
import pandas as pd
import joblib
import ipdb
from encoders import ImageScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from keras.models import model_from_json
from tensorflow import keras
from keras.optimizers import Adam
from PIL import Image

# class Preprocessor(object):

#     @classmethod
#     def from_path(cls, model_dir="."):
#         # model_path = os.path.join(model_dir, 'pipeline.joblib')
#         model = joblib.load("./pipeline.joblib")
#         return cls(model)

#     def __init__(self):
#         # ipdb.set_trace()
#         self.pipeline = Preprocessor.from_path()

#     def predict(self, df):
#         try:
#             preproc_instances = self.pipeline.predict(df)
#             return preproc_instances
#         except BaseException:
#             return {"error": True, "traceback": traceback.format_exc()}

## Load up model

met_test = pd.DataFrame({
          'acral': float(0),
          'abdomen' : float(0),
          'back' : float(0),
          'chest' : float(0),
          'ear' : float(1),
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
          'confocal': float(1),
          'consensus': float(0),
          'follow_up': float(0),
          'histo': float(0),
          'female': float(1),
          'male': float(0),
          'unknown': float(0),
          'age_scaled': (float(50)-60)/25
          }, index=[0])

## import image

colourImg = Image.open("../dataset/HAM10000_images_part_1/ISIC_0024306.jpg")
colourPixels = colourImg.convert("RGB")
colourArray = np.array(colourPixels.getdata())
test_pic = colourArray.reshape((450, 600, 3))
test_im = np.resize(test_pic, (1,75, 100, 3))
print(test_im.shape)

## load model

name='vgg_test'
json_file = open(f'{name}', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(f'{name}.h5')
model = loaded_model

# predict using model

results = model.predict(x=[met_test])
print(results)

# opt = Adam(lr=1e-3, decay=1e-3 / 200)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# model.predict(x=[X_im_test, X_met_test])

# print("--------------- MODEL LOADED ---------------------")



