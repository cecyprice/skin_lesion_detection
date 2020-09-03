
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse


class ImageScaler(BaseEstimator, TransformerMixin):
  def __init__(self, scaler='normalization', image_size='full_size'):
    self.scaler=scaler
    self.image_size=image_size

  def transform(self, X, y=None):
    # if transforming full_size images
    # if self.image_size=='full_size':
    #   if self.scaler=='normalization':
    #     X['pixels_scaled'] = X.images.apply(lambda x: x/255)
    #   if self.scaler=='standardization':
    #     scaler = StandardScaler()
    #     X['pixels_scaled'] = X.images.apply(lambda x: (x - x.mean(axis=0))/x.std(axis=0))
    #   if self.scaler=='centering':
    #     X['pixels_scaled'] = X.images.apply(lambda x: ((x - x.mean(axis=0))-(x - x.mean(axis=0)).min())/((x - x.mean(axis=0)).max()-(x - x.mean(axis=0)).min()))

    # # if transforming resied images
    # elif self.image_size=='resized':
    #   if self.scaler=='normalization':
    #     X['pixels_scaled'] = X.images_resized.apply(lambda x: x/255)
    #   if self.scaler=='standardization':
    #     scaler = StandardScaler()
    #     X['pixels_scaled'] = X.images_resized.apply(lambda x: (x - x.mean(axis=0))/x.std(axis=0))
    #   if self.scaler=='centering':
    #     X['pixels_scaled'] = X.images_resized.apply(lambda x: ((x - x.mean(axis=0))-(x - x.mean(axis=0)).min())/((x - x.mean(axis=0)).max()-(x - x.mean(axis=0)).min()))

    # rows = X.pixels_scaled.values.shape[0]
    # return X.pixels_scaled.values.reshape(rows, 1)

    return self.values




  def fit(self, X, y=None):
    return self

if __name__ == "__main__":
  print('encoded features')

