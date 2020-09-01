import os
import traceback
import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from skin_lesion_detection.encoders import ImageScaler



class Preprocessor(object):

    @classmethod
    def from_path(cls, model_dir="."):
        # model_path = os.path.join(model_dir, 'pipeline.joblib')
        model = joblib.load("./pipeline.joblib")
        return cls(model)

    def __init__(self):
        self.pipeline = Preprocessor.from_path()

    def predict(self, df):
        try:
            preproc_instances = self.pipeline.predict(df)
            return preproc_instances
        except BaseException:
            return {"error": True, "traceback": traceback.format_exc()}







