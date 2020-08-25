import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

    ## Define random image modifications
    aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

    #construct the actual Python generator
    dataGen = aug.flow(df,
                       batch_size = df.shape[0])

    ## OPTIONAL BIT: IF YOU WANT TO SAVE NEW IMAGES INTO ORIGINAL FOLDER## 

    '''uncomment start
    # output = 'YOUR_FILEPATH_HERE'
    # dataGen = aug.flow(df,
    #                    batch_size = df.shape[0],
    #                    save_to_dir = output)
    uncomment finish '''

    # iterate over imagegenerator object, add to test_images dataset
    for i in dataGen:
        break

    ## Entire dataset doubled

    df = np.concatenate((df, i), axis = 0)

    return df

if __name__ == '__main__':
  print('cleaned dataframe')


# ## ap = argparse.ArgumentParser()
#   ap.add_argument("-i", "--image", required=True,
#     help="path to the input image")
#   ap.add_argument("-o", "--output", required=True,
#     help="path to output directory to store augmentation examples")
#   ap.add_argument("-t", "--total", type=int, default=100,
#     help="# of training samples to generate")
#   args = vars(ap.parse_args())
