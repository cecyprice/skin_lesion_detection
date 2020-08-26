import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from glob import glob
import matplotlib.pyplot as plt
import imageio
from PIL import Image


def get_data(n_rows=10000, random_state=1, **kwargs):
  '''
  Import and merge dataframes, pass n_rows arg to pd.read_csv to get a sample dataset
  '''

  base_skin_dir = os.path.join('..','dataset')
  imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

  lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
  }

  df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

  df['path'] = df['image_id'].map(imageid_path_dict.get)
  df['cell_type'] = df['dx'].map(lesion_type_dict.get)
  df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

  df['path'].dropna(inplace=True)

  df['images'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75)))).apply(lambda x : x.reshape(22500))

  return df


def clean_df(df):
  ## fill missing values with mean age
  df['age'].fillna((df['age'].mean()), inplace = True)

  ## drop duplicates
  df = df.drop_duplicates(subset=['lesion_id'], keep = 'first')

  return df

def balance_nv(df, under_sample_size):

        ## isolate nv rows
        data_nv = df[df['dx'] == 'nv']

        # define scaling parameters
        sample_size = under_sample_size
        scaling = under_sample_size / data_nv.shape[0]

        # stratified sampling
        rus = RandomUnderSampler(sampling_strategy={'lower extremity' : int(1224*scaling),
                                                    'trunk' : int(1153*scaling),
                                                    'back' : int(1058*scaling),
                                                    'abdomen' : int(719*scaling),
                                                    'upper extremity' : int(504*scaling) ,
                                                    'foot' : int(209*scaling),
                                                    'unknown' : int(175*scaling),
                                                    'chest' : int(112*scaling),
                                                    'face' : int(61*scaling),
                                                    'neck' : int(60*scaling),
                                                    'genital' : int(43*scaling),
                                                    'hand' : int(39*scaling),
                                                    'scalp' : int(24*scaling),
                                                    'ear' : int(19*scaling),
                                                    'acral' : int(3*scaling)+1
                                                   },
                                   random_state=None,
                                   replacement=False,
                                )

        ## fit strtaified sampling model
        n_x, n_y = rus.fit_resample(data_nv, data_nv['localization'])

        ## delete nv rows from original dataset
        no_nv_data = df[df.dx != 'nv']

        df = pd.concat([n_x, no_nv_data], axis=0)

        return df

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

