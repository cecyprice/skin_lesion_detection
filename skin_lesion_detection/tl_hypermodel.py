import warnings
from termcolor import colored
import numpy as np
import os
from tensorflow import keras
import tensorflow.keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Input, concatenate, BatchNormalization, Dense, Dropout, Activation, Flatten, Embedding, Conv1D, Conv2D, MaxPooling2D, MaxPool1D
import kerastuner
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel
from kerastuner.tuners.randomsearch import RandomSearch


from tl_models import TLModels
from data import get_data, clean_df, balance_nv, data_augmentation
from trainer import Trainer



class TLRegressionHyperModel(HyperModel):
  """
  Build HyperModel allowing hyperparamter tuning
  """
  def __init__(self, input_dim, input_shape, selection='vgg16'):
    self.input_dim = input_dim
    self.input_shape = input_shape
    self.num_labels = t.num_labels
    self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    self.selection = selection


  def build(self, hp):
    # mlp fork
    self.mlp_fork = Sequential()
    self.mlp_fork.add(Dense(units=hp.Int('units', min_value=8, max_value=512, step=32, default=128),
                        input_dim=self.input_dim,
                        activation="relu"))
    self.mlp_fork.add(Dense(4, activation="relu"))

    # cnn fork
    if self.selection == 'vgg16':
            model = VGG16(weights='imagenet',
                          input_shape=self.input_shape,
                          include_top=False)

    if self.selection == 'resnet':
        model = ResNet50(weights='imagenet',
                         input_shape=self.input_shape,
                         include_top=False)

    if self.selection == 'densenet':
        model = DenseNet121(weights='imagenet',
                            input_shape=self.input_shape,
                            include_top=False)

    for layer in model.layers:
        layer.iterable = False
        layer.trainable = False

    inp = Input(shape=self.input_shape)

    base_output = model(inp)
    x = Flatten()(base_output)
    x = Dense(units=hp.Int('units', min_value=8, max_value=512, step=32, default=128),
            activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'], default='relu'))(x)
    x = Dropout(0.5)(x)x = Dense(4, activation='relu')(x)

    self.cnn_fork = Model(inp, x)

    # combine MLP and CNN forks to create merged predictor
    combinedInput = concatenate([self.mlp_fork.output, self.cnn_fork.output])
    out = Dense(4, activation="relu")(combinedInput)
    out = Dense(self.num_labels, activation="softmax")(out)
    merged_model = Model(inputs=[self.mlp_fork.input, self.cnn_fork.input], outputs=out)

    merged_model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-3)),
                        loss="categorical_crossentropy",
                        metrics=['accuracy'])

    return merged_model


if __name__ == "__main__":

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Get and clean data
    image_size = 'resized' # toggle between 'resized' and 'full_size'
    df = get_data(nrows=100)
    print("-----------STATUS UPDATE: DATA IMPORTED'-----------")
    df = clean_df(df)
    print("-----------STATUS UPDATE: DATA CLEANED'-----------")
    #df = balance_nv(df, 1000)
    #df = data_augmentation(df, image_size=image_size)
    print("-----------STATUS UPDATE: DATA BALANCED + AUGMENTED'-----------")

    # Assign X and y and instanciate Trainer Class
    X = df.drop(columns=['dx', 'lesion_id', 'image_id', 'cell_type', 'cell_type_idx'])
    y = df['dx']
    t = Trainer(X, y, image_size=image_size)

    # Preprocess data: transfrom and scale
    print(colored("############  Preprocessing data   ############", 'blue'))
    t.preprocess()

    # Seach hyperparamaters for optimal values
    print(colored("############  Tuning hyperparamaters   ############", 'blue'))
    hypermodel = TLRegressionHyperModel(input_dim=t.input_dim, input_shape=t.input_shape, selection='vgg16')

    tuner = RandomSearch(hypermodel,
                objective='val_accuracy',
                seed=42,
                max_trials=10,
                executions_per_trial=2,
                directory=os.path.normpath('C:/'))

    tuner.search(x=[t.X_met_train, t.X_im_train], y=t.y_train,
                validation_split=0.3,
                epochs=5,
                verbose=1,
                callbacks=[hypermodel.es])

    # import ipdb; ipdb.set_trace()
    best_model = tuner.get_best_models(num_models=1)[0]

    print(colored(f"Best Model: {best_model}", 'green'))

    # Evaluate with test data
    print(colored("############  Tuning hyperparamaters   ############", 'blue'))
    search_results = best_model.evaluate(x=[t.X_met_test, t.X_im_test], y=t.y_test)
    print(colored(f"Evaluation results: {search_results}", 'green'))
