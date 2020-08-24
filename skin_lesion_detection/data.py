import pandas as pd
import numpy as np


def get_data(n_rows=10000, random_state=1, **kwargs):
  '''
  Import and merge dataframes, pass n_rows arg to pd.read_csv to get a sample dataset
  '''
  pass


def clean_df(df):
  '''
  Dropna, remove duplicates
  '''
  pass


def optimise_df(df, verbose=True, **kwargs):
  '''
  Reduce size of dataframe by downcasting numerical columns
  '''
  pass


def data_augmentation(df):
  '''
  Generate more images through data augmentation(rotation, height_shift, width_shift, ...)
  Only generate images of the under-represented classes ???
  '''
  pass




if __name__ == '__main__':
  print('cleaned dataframe')
