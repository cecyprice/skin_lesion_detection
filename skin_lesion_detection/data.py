import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from glob import glob
import matplotlib.pyplot as plt
import sklearn.neighbors
from imblearn.under_sampling import RandomUnderSampler
import imageio
from PIL import Image


def get_data(random_state=1, nrows=None):
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

  df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'), nrows=nrows)

  df['path'] = df['image_id'].map(imageid_path_dict.get)
  df['cell_type'] = df['dx'].map(lesion_type_dict.get)
  df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

  df['path'].dropna(inplace=True)

  df['images'] = df['path'].map(lambda x: np.asarray(Image.open(x))).apply(lambda x : x.reshape(810000))
  df['images_resized'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75)))).apply(lambda x : x.reshape(22500))

  return df


def clean_df(df):
  ## fill missing values with mean age
  df['age'].fillna((df['age'].mean()), inplace = True)

  ## drop duplicates
  df = df.drop_duplicates(subset=['lesion_id'], keep = 'first')

  # convert categorical columns to numeric values
  df.localization = df.localization.astype('category')
  df.dx_type = df.dx_type.astype('category')
  df.sex = df.sex.astype('category')

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

    ## Create np.array of augmented images from original images dataframe. Reshape to feed into dataGen
    images_array = np.array([i.reshape(75,100,3) for i in df['images_resized'].values])

    #construct the actual Python generator, iterate over imagegenerator object
    dataGen = aug.flow(images_array, batch_size = len(df))
    for i in dataGen:
        break

    ## flatten i before concatenating it into new dataframe copy
    i = i.reshape(len(df), 22500)

    ## turn i from array into list so it can be converted into pd
    im_list = []
    for im in i:
        im_list.append(im)

    # convert i into the pandas i_df
    i_df = pd.DataFrame({'images_resized': im_list})

    # create new dataframe without image column and convert in np.array
    new_df = df.loc[:, df.columns != 'images_resized']

    ## concatenate new_df numpy array and new augmented image array
    new_df = pd.concat((new_df, i_df), axis = 1)

    ## convert new_df back into pandas
    new_df = pd.DataFrame(new_df)

    ## vertically concatenate new dataframes
    frames = [df, new_df]
    df = pd.concat(frames)
    df.reset_index(drop=True, inplace=True)

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

