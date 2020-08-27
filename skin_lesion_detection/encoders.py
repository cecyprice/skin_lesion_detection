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
  def __init__(self, scaler='normalization',image_size='full_size'):
    self.scaler=scaler
    self.image_size=image_size

  def transform(self, X, y=None):
    if self.scaler=='normalization':
      X['pixels_scaled'] = X.image_size.apply(lambda x: x/255)
    if self.scaler=='standardization':
      scaler = StandardScaler()
      X['pixels_scaled'] = X.image_size.apply(lambda x: (x - x.mean(axis=0))/x.std(axis=0))
    if self.scaler=='centering':
      X['pixels_scaled'] = X.image_size.apply(lambda x: ((x - x.mean(axis=0))-(x - x.mean(axis=0)).min())/((x - x.mean(axis=0)).max()-(x - x.mean(axis=0)).min()))

    return X[['pixels_scaled']]

  def fit(self, X, y=None):
    return self

if __name__ == "__main__":
  print('encoded features')

