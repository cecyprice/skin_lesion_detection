import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



class LabelEncoder():

  def __init__(self, encoder):
    self.encoder = encoder


  def transform(self):
    pass


  def fit(self, X, y=None):
    return self




if __name__ == "__main__":
  print('encoded features')
