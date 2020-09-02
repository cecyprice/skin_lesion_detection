import os
import traceback
import pandas as pd
import joblib
import ipdb
from skin_lesion_detection.encoders import ImageScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler




class Preprocessor(object):

    @classmethod
    def from_path(cls, model_dir="."):
        # model_path = os.path.join(model_dir, 'pipeline.joblib')
        model = joblib.load("./pipeline.joblib")
        return cls(model)

    def __init__(self):
        # ipdb.set_trace()
        self.pipeline = Preprocessor.from_path()

    def predict(self, df):
        try:
            preproc_instances = self.pipeline.predict(df)
            return preproc_instances
        except BaseException:
            return {"error": True, "traceback": traceback.format_exc()}







