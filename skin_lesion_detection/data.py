import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data(n_rows=10000, random_state=1, **kwargs):
  '''
  Import and merge dataframes, pass n_rows arg to pd.read_csv to get a sample dataset
  '''
  path = '~/code/cecyprice/skin_lesion_detection/dataset/'
  dim1_L = pd.read_csv(path + 'hmnist_8_8_L.csv')
  dim1_RGB = pd.read_csv(path + 'hmnist_8_8_RGB.csv')
  dim2_L = pd.read_csv(path + 'hmnist_28_28_L.csv')
  dim2_RGB = pd.read_csv(path + 'hmnist_28_28_RGB.csv')

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

  skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

  skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
  skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
  skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

  skin_df['hmnist_8_8_L'] = dim1_L.apply(lambda r: tuple(r), axis=1).apply(np.array)
  skin_df['hmnist_8_8_RGB'] = dim1_RGB.apply(lambda r: tuple(r), axis=1).apply(np.array)
  skin_df['hmnist_28_28_L'] = dim2_L.apply(lambda r: tuple(r), axis=1).apply(np.array)
  skin_df['hmnist_28_28_RGB'] = dim2_RGB.apply(lambda r: tuple(r), axis=1).apply(np.array)

  skin_df['image_100_75'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
  skin_df['image_100_75_reshaped'] = skin_df['image_100_75'].apply(lambda x : x.reshape(22500))

  return skin_df


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
