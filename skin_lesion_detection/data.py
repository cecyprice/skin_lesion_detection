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

from params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, PROJECT_ID
# from google.cloud import storage

def get_data(random_state=1, local=True, nrows=None):
  '''
  Import and merge dataframes, pass n_rows arg to pd.read_csv to get a sample dataset
  '''
  lesion_type_dict = {
       'nv': 'Melanocytic nevi',
       'mel': 'Melanoma',
       'bkl': 'Benign keratosis-like lesions ',
       'bcc': 'Basal cell carcinoma',
       'akiec': 'Actinic keratoses',
       'vasc': 'Vascular lesions',
       'df': 'Dermatofibroma'
  }

  if local:
    base_skin_dir = os.path.join('..','dataset')
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                      for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

    df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'), nrows=nrows)

    df['path'] = df['image_id'].map(imageid_path_dict.get)
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

    ## fill missing values with mean age
    df['age'].fillna((df['age'].mean()), inplace = True)
    df = df.dropna()

    df['images'] = df['path'].map(lambda x: np.asarray(Image.open(x))).apply(lambda x : x.reshape(810000))
    df['images_resized'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((75,100)))).apply(lambda x : x.reshape(22500))

    return df

  else:

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.get_blob('dataset/HAM10000_metadata.csv')
    csv_name = blob.name.split("/")[1]
    blob.download_to_filename(csv_name)
    df = pd.read_csv(csv_name, nrows=nrows)
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

    dict_img = {'image_id': [], 'images': []}
    blobs=list(bucket.list_blobs())

    for blob in blobs:
      blob_name = blob.name
      if ".jpg" in blob_name:
        img_name = blob_name.split("/")[2]
        blob_0 = bucket.blob(blob_name)
        blob_0.download_to_filename(img_name)
        img_0 = np.asarray(Image.open(img_name))
        id = img_name.split(".")[0]
        dict_img['image_id'].append(id)
        dict_img['images'].append(img_0.reshape(810000))

    df2 = pd.DataFrame.from_dict(dict_img)
    df3 = df2.merge(df, how='left', on='image_id')

    return df3


def clean_df(df):

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


def data_augmentation(df, image_size = 'resized'):
    print(df.shape)
    df = df.reset_index(drop=True)
    ## Define random image modifications
    aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

    if image_size == 'resized':
        target_images = 'images_resized'
        input_size = (75,100,3)
        df = df.drop(['images'], axis =1)
        new_df = df.copy()
        new_df = new_df.drop(['images_resized'], axis=1)


    elif image_size == 'full_size':
        target_images = 'images'
        input_size = (450,600,3)
        df = df.drop(['images_resized'], axis =1)
        new_df = df.copy()
        new_df = new_df.drop(['images'], axis=1)

    ## Create np.array of augmented images from original images dataframe. Reshape to feed into dataGen
    images_array = np.array([i.reshape(input_size) for i in df[target_images].values])

    #construct the actual Python generator, iterate over imagegenerator object
    dataGen = aug.flow(images_array, batch_size = images_array.shape[0])
    for i in dataGen:
        break

    ## flatten i before concatenating it into new dataframe copy
    i = i.reshape(len(df), input_size[0]*input_size[1]*input_size[2])

    ## turn i from array into list so it can be converted into pd
    im_list = []
    for im in i:
        im_list.append(im)

    # convert i into the pandas i_df
    i_df = pd.DataFrame({target_images: im_list})

    ## concatenate new_df numpy array and new augmented image array
    com_new_df = pd.concat((new_df, i_df), axis = 1)

    ## vertically concatenate new dataframes

    frames = [df, com_new_df]
    df = pd.concat(frames)
    df.reset_index(drop=True, inplace=True)
    print(df.shape)
    return df


if __name__ == '__main__':

  print('cleaned dataframe')
  get_data()
  print(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'dataset'))



# ## ap = argparse.ArgumentParser()
#   ap.add_argument("-i", "--image", required=True,
#     help="path to the input image")
#   ap.add_argument("-o", "--output", required=True,
#     help="path to output directory to store augmentation examples")
#   ap.add_argument("-t", "--total", type=int, default=100,
#     help="# of training samples to generate")
#   args = vars(ap.parse_args())

