import warnings
from termcolor import colored
import numpy as np
import os
import tensorflow.keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, concatenate, BatchNormalization, Dense, Dropout, Activation, Flatten, Embedding, Conv1D, Conv2D, MaxPooling2D, MaxPool1D
import kerastuner
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel
from kerastuner.tuners.randomsearch import RandomSearch

from transfer_learning_models import TLModels
from data import get_data, clean_df, balance_nv, data_augmentation
from trainer import Trainer



class RegressionHyperModel(HyperModel):
  """
  Build HyperModel allowing hyperparamter tuning
  """
  def __init__(self, input_dim, input_shape):
    self.input_dim = input_dim
    self.input_shape = input_shape
    self.num_labels = t.num_labels
    self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


  def build(self, hp):
    # mlp fork
    mlp_fork = Sequential()
    mlp_fork.add(Dense(units=hp.Int('units', min_value=8, max_value=512, step=32, default=128),
                        input_dim=self.input_dim,
                        activation="relu"))
    mlp_fork.add(Dense(4, activation="relu"))

    # cnn fork
    chanDim = -1
    inputs = Input(shape=self.input_shape)

    for i in range(3):
        if i == 0:
            x = inputs
        x = Conv2D(filters=hp.Choice('num_filters', values=[16, 32, 64], default=64),
                  kernel_size=hp.Choice('kernel_size', values=[2, 3, 4, 5], default=3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(units=hp.Int('units', min_value=8, max_value=512, step=32, default=128),
              activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'], default='relu'))(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    x = Dense(4)(x)
    x = Activation("relu")(x)

    cnn_fork = Model(inputs, x)


    # combine MLP and CNN forks to create merged predictor
    combinedInput = concatenate([mlp_fork.output, cnn_fork.output])

    out = Dense(4, activation="relu")(combinedInput)
    out = Dense(self.num_labels, activation="softmax")(out)
    merged_model = Model(inputs=[mlp_fork.input, cnn_fork.input], outputs=out)

    # compile the model using BCE as loss and Adam otpimizer with varied values
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
    hypermodel = RegressionHyperModel(input_dim=t.input_dim, input_shape=t.input_shape)

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

    best_model = tuner.get_best_models(num_models=1)[0]

    print(colored(f"Best Model: {best_model}", 'green'))


    # Evaluate with test data
    print(colored("############  Tuning hyperparamaters   ############", 'blue'))
    search_results = best_model.evaluate(x=[t.X_met_test, t.X_im_test], y=t.y_test)
    print(colored(f"Evaluation results: {search_results}", 'green'))
