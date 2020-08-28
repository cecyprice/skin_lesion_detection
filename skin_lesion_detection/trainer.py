from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler

import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping

from baseline_model import BaselineModel
from transfer_learning_models import TLModels
from data import get_data, clean_df, balance_nv, data_augmentation
from encoders import ImageScaler

import pandas as pd
import numpy as np
import warnings

class Trainer(object):

    def __init__(self, X, y, **kwargs):

        self.pipeline = None
        self.kwargs = kwargs
        self.X = X
        self.y = y
        self.split = self.kwargs.get("split", True)

        # Image dimension attributes
        self.scaler = self.kwargs.get('scaler', 'normalization')
        self.image_size = self.kwargs.get('image_size', 'full_size')
        if self.image_size == 'full_size':
          self.target_images = 'images'
          self.input_shape = (450, 600, 3)
        elif self.image_size == 'resized':
          self.target_images = 'images_resized'
          self.input_shape = (75, 100, 3)



    def get_estimator(self):
        # get different models as self.model
        if self.estimator=='baseline_model':
            self.model = BaselineModel().merge_compile_models(input_dim=self.input_dim, input_shape=self.input_shape, num_labels=self.num_labels)
        elif self.estimator=='tl_vgg':
            self.model = TLModels().tl_merge_compile_models(input_dim=self.input_dim, input_shape=self.input_shape, selection='vgg16', num_labels=self.num_labels)
        elif self.estimator=='tl_resnet':
            self.model = TLModels().tl_merge_compile_models(input_dim=self.input_dim, input_shape=self.input_shape, selection='resnet', num_labels=self.num_labels)
        elif self.estimator=='tl_densenet':
            self.model = TLModels().tl_merge_compile_models(input_dim=self.input_dim, input_shape=self.input_shape, selection='densenet', num_labels=self.num_labels)


    def set_pipeline(self):

        # Define feature engineering pipeline blocks
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.rs = RobustScaler()
        self.imsc = ImageScaler(scaler=self.scaler, image_size=self.image_size)

        pipe_cat_feats = make_pipeline(self.ohe)
        pipe_cont_feats = make_pipeline(self.rs)
        pipe_photo_feats = make_pipeline(self.imsc)

        # Define default feature engineering blocs
        feateng_blocks = [
            ('cat_feats', pipe_cat_feats, ['localization', 'dx_type', 'sex']),
            ('cont_features', pipe_cont_feats, ['age']),
            ('photo_feats', pipe_photo_feats, [self.target_images]),
        ]

        self.features_encoder = ColumnTransformer(feateng_blocks, n_jobs=None, remainder="drop")

        self.pipeline = Pipeline(steps=[
            ('features', self.features_encoder)
            ])


    def add_grid_search(self):
        """"
        Apply Gridsearch on self.params defined in get_estimator - using RegressionHyperModel?
        """
        pass


    #@simple_time_tracker
    def preprocess(self, gridsearch=False, image_type="full_size"):
        """
        Add time tracker - if we want?
        """
        # categorise y
        ohe = OneHotEncoder(handle_unknown='ignore')
        self.num_labels = len(np.unique(self.y.values))
        self.y = ohe.fit_transform(self.y.values.reshape(-1, 1)).toarray()
        print("-----------STATUS UPDATE: Y CATEGORISED'-----------")

        # convert x categorical features to strings
        self.X['localization'] = self.X['localization'].to_string()
        self.X['dx_type'] = self.X['dx_type'].to_string()
        self.X['sex'] = self.X['sex'].to_string()
        self.X['age'] = self.X['age'].astype('float64')

        # scale/encode X features (metadata + pixel data) via pipeline
        self.set_pipeline()
        self.X = self.pipeline.fit_transform(self.X)

        # convert self.X to pd.df
        self.col_list = []
        list_arrays = self.features_encoder.transformers_[0][1].named_steps['onehotencoder'].categories_
        for i in list_arrays:
            for col_name in i:
                self.col_list.append(col_name)
        self.col_list.append('age_scaled')
        self.col_list.append('pixels_scaled')

        self.X = pd.DataFrame(self.X, columns=self.col_list)
        print("-----------STATUS UPDATE: PIPELINE FITTED'-----------")

        # create train vs test dataframes
        if self.split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=1, test_size=0.3)

        self.pixels_to_array(image_type=self.image_size)
        print("-----------STATUS UPDATE: DATA SPLIT INTO X/Y TEST/TRAIN MET/IM'-----------")


    def pixels_to_array(self, image_type="full_size"):
        """
        Convert X_train and X_test into [X_met_train + X_im_train] and [X_met_test + X_im_test] respectively
        """
        self.X_met_train = self.X_train.drop(columns=['pixels_scaled']).astype('float64')
        self.X_met_test = self.X_test.drop(columns=['pixels_scaled']).astype('float64')

        if image_type == "full_size":
            self.X_im_train = np.array([i.reshape(450, 600, 3) for i in self.X_train['pixels_scaled'].values])
            self.X_im_test = np.array([i.reshape(450, 600, 3) for i in self.X_test['pixels_scaled'].values])
        elif image_type == "resized":
            self.X_im_train = np.array([i.reshape(75, 100, 3) for i in self.X_train['pixels_scaled'].values])
            self.X_im_test = np.array([i.reshape(75, 100, 3) for i in self.X_test['pixels_scaled'].values])
        print("-----------STATUS UPDATE: PIXEL ARRAGYS EXTRACTED'-----------")


    #@simple_time_tracker
    def train(self, gridsearch=False, estimator='baseline_model'):
        # assign self.estimator as desired estimator and set self.model via get_estimator()
        self.input_dim=self.X_met_train.shape[1]
        self.estimator=estimator
        self.get_estimator()

        # define es criteria and fit model
        es = EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1, restore_best_weights=True)
        self.history = self.model.fit(x=[self.X_met_train, self.X_im_train], y=self.y_train,
            validation_split=0.3,
            epochs=200,
            callbacks = [es],
            batch_size=8,
            verbose = 1)


    def evaluate(self):
        ## SEE TRAINING MODEL ACCURACY
        self.train_results = self.model.evaluate(x=[self.X_met_train, self.X_im_train], y=self.y_train, verbose=1)
        print('Train Loss: {} - Train Accuracy: {}'.format(self.train_results[0], self.train_results[1]))
        # print('Train Loss: {} - Train Accuracy: {} - Train Recall: {} - Train Precision: {}'.format(self.train_met_results[0], self.train_met_results[1], train_met_results[2], train_met_results[3]))

        ## TEST DATA ACCURACY
        self.test_results = self.model.evaluate(x=[self.X_met_test, self.X_im_test], y=self.y_test, verbose=1)
        print('Test Loss: {} - Test Accuracy: {}'.format(self.test_results[0], self.test_results[1]))
        # print('Test Loss: {} - Test Accuracy: {} - Test Recall: {} - Test Precision: {}'.format(test_met_results[0], test_met_results[1], test_met_results[2], test_met_results[3]))


    def plot_loss_accuracy(history):

        fig, axs = plt.subplots(2)

        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.xlabel("Epochs")
        plt.legend(['Train', 'val_test'], loc='best')

        axs[1].plot(history.history['accuracy'])
        axs[1].plot(history.history['val_accuracy'])
        plt.title("Model Accuracy")
        plt.xlabel("Epochs")
        plt.legend(['Train', 'val_test'], loc='best')


    def save_model(self):
        """
        Save the model into a .joblib
        """
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))
        pass


    # ### MLFlow methods
    # @memoized_property
    # def mlflow_client(self):
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     try:
    #         return self.mlflow_client.create_experiment(self.experiment_name)
    #     except BaseException:
    #         return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    # @memoized_property
    # def mlflow_run(self):
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     if self.mlflow:
    #         self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     if self.mlflow:
    #         self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    # def log_estimator_params(self):
    #     reg = self.get_estimator()
    #     self.mlflow_log_param('estimator_name', reg.__class__.__name__)
    #     params = reg.get_params()
    #     for k, v in params.items():
    #         self.mlflow_log_param(k, v)

    # def log_kwargs_params(self):
    #     if self.mlflow:
    #         for k, v in self.kwargs.items():
    #             self.mlflow_log_param(k, v)

    # def log_machine_specs(self):
    #     cpus = multiprocessing.cpu_count()
    #     mem = virtual_memory()
    #     ram = int(mem.total / 1000000000)
    #     self.mlflow_log_param("ram", ram)
    #     self.mlflow_log_param("cpus", cpus)


if __name__ == "__main__":

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Get and clean data
    df = get_data(nrows=None)
    print("-----------STATUS UPDATE: DATA IMPORTED'-----------")
    df = clean_df(df)
    print("-----------STATUS UPDATE: DATA CLEANED'-----------")
    df = balance_nv(df, 1000)
    "-----------STATUS UPDATE: DATA BALANCED-----------"
    df = data_augmentation(df, image_size='resized')
    print("-----------STATUS UPDATE: DATA AUGMENTED'-----------")

    # Assign X and y and instanciate Trainer Class
    X = df.drop(columns=['dx', 'lesion_id', 'image_id', 'cell_type', 'cell_type_idx'])
    y = df['dx']
    t = Trainer(X, y, image_size='resized')

    # Preprocess data: transfrom and scale
    print("############  Preprocessing data   ############")
    t.preprocess(image_type=t.image_size)

    # Train model
    print("############  Training model   ############")
    t.train(estimator='tl_vgg')

    # Evaluate model on X_test/y_preds vs y_test
    print("############  Evaluating model   ############")
    t.evaluate()

    # Plot history
    # plot_loss_accuracy(self.history)






## Matt qs:
        ## should we write an evaluate function? model.evalute for cnns
        ## if we are writing evaluate function: y_test/pred = 7 column OHE matrix
        ## either: convert back to classes/0-6 numbers OR map one matrix onto another (TRUE/FALSE) and
        ## take number of rows containing FALSE / total number of rows
