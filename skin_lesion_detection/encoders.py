import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

class LabelEncoder():

  def __init__(self, encoder):
    self.encoder = encoder


  def transform(self):
    pass


  def fit(self, X, y=None):
    return self


class ImageScaler():
  def __init__(self, scaler='normalization',**kwargs):
    self.scaler=scaler

  def transform(self, df):
    if self.scaler=='normalization':
      df['images_scaled'] = df.images.apply(lambda x: x/255)
    if self.scaler=='standardization':
      scaler = StandardScaler()
      df['images_scaled'] = df.images.apply(lambda x: (x - x.mean(axis=0))/x.std(axis=0))
    if self.scaler=='centering':
      df['images_scaled'] = df.images.apply(lambda x: ((x - x.mean(axis=0))-(x - x.mean(axis=0)).min())/((x - x.mean(axis=0)).max()-(x - x.mean(axis=0)).min()))

    return df[['images_scaled']]

  def fit(self, X, y=None):
    return self

if __name__ == "__main__":
  print('encoded features')

