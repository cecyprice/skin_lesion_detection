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

  def transform(self, X, y=None):
    assert isinstance(X, pd.DataFrame)
    if self.scaler=='normalization':
      X['images_scaled'] = X.images.apply(lambda x: x/255)
    if self.scaler=='standardization':
      scaler = StandardScaler()
      X['images_scaled'] = X.images.apply(lambda x: (x - x.mean(axis=0))/x.std(axis=0))
    if self.scaler=='centering':
      X['images_scaled'] = X.images.apply(lambda x: ((x - x.mean(axis=0))-(x - x.mean(axis=0)).min())/((x - x.mean(axis=0)).max()-(x - x.mean(axis=0)).min()))

    return X[['images_scaled']]

  def fit(self, X, y=None):
    return self

if __name__ == "__main__":
  print('encoded features')

