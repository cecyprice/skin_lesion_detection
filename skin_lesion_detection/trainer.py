from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from tensorflow.keras import EarlyStopping

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
        self.history = history
        self.train_met_results = train_met_results
        self.train_img_results = train_img_results
        self.test_met_results = test_met_results
        self.test_img_results = test_img_results


    def get_estimator(self):
        # get mixed model as self.mixed_model
        pass

    def set_pipeline(self):

        # Define feature engineering pipeline blocks
        pipe_cat_feats = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
        pipe_cont_feats = make_pipeline(RobustScaler())
        # pipe_photo_feats = make_pipeline(CUSTOMSCALERFORPIXELDATA())


        # Define default feature engineering blocs
        feateng_blocks = [
            ('cat_feats', pipe_cat_feats, ['localization', 'dx_type', 'sex']),
            ('cont_features', pipe_cont_feats, ['age'])
            # ('photo_feats', pipe_photo_feats, LISTOFPIXELCOLUMNS),
        ]

        features_encoder = ColumnTransformer(feateng_blocks, n_jobs=None, remainder="drop")

        self.pipeline = Pipeline(steps=[
            ('features', features_encoder)
            ])


    def add_grid_search(self):
        """"
        Apply Gridsearch on self.params defined in get_estimator - using RegressionHyperModel?
        """
        pass


    #@simple_time_tracker
    def preprocess(self, gridsearch=False):
        """
        Add time tracker - if we want?
        """
        # categorise y
        ohe = OneHotEncoder(handle_unknown='ignore')
        self.y = ohe.fit_transform(self.y.values.reshape(-1, 1))

        # scale/encode X features (metadata + pixel data) via pipeline
        self.set_pipeline()
        self.pipeline.fit_transform(self.X, self.y)

        # create train vs test dataframes
        if self.split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=1, test_size=0.3)

        # Convert X_train and X_test into [X_met_train + X_im_train] and [X_met_test + X_im_test] respectively
        self.X_met_train = self.X_train[['age', 'sex', 'dx_type', 'localization']]
        self.X_im_train = np.array([i.reshape(75, 100, 3) for i in self.X_train['images_resized'].values])
        self.X_met_test = self.X_train[['age', 'sex', 'dx_type', 'localization']]
        self.X_im_test = np.array([i.reshape(75, 100, 3) for i in self.X_test['images_resized'].values])


    #@simple_time_tracker
    def train_predict(self, gridsearch=False):

        es = EarlyStopping(monitor="val_loss", mode="auto", patience=50)

        model = self.mixed_model #from mixed_model.py

        self.history = model.fit(x=[self.X_met_train, self.X_im_train], y=self.y_train,
        validation_split=0.3,
        epochs=200,
        callbacks = [es],
        batch_size=8,
        verbose = 1)

        self.y_preds = model.predict([self.X_met_test, self.X_im_test])


## 'accuracy', 'recall', 'precision', 'f1'

    def evaluate(self):

      ## SEE TRAINING MODEL ACCURACY
      self.train_met_results = model.evaluate(self.X_met_train, self.y_train, verbose=0)
      print('Train Meta loss: {} - Train Meta Accuracy: {} - Train Meta Recall: {} - Train Meta Precision: {}'.format(train_met_results[0], train_met_results[1], train_met_results[2], train_met_results[3]))

      self.train_img_results = model.evaluate(self.X_im_train, self.y_train, verbose=0)
      print('Train Image loss: {} - Train Image Accuracy: {} - Train Image Recall: {} - Train Image Precision: {}'.format(train_img_results[0], train_img_results[1], train_img_results[2], train_img_results[3]))

      ## TEST DATA ACCURACY

      self.test_met_results = model.evaluate(self.X_met_test, self.y_test, verbose=0)
      print('Test Meta loss: {} - Test Meta Accuracy: {} - Test Meta Recall: {} - Test Meta Precision: {}'.format(test_met_results[0], test_met_results[1], test_met_results[2], test_met_results[3]))

      self.test_img_results = model.evaluate(self.X_im_train, self.y_test, verbose=0)
      print('Test Image loss: {} - Test Image Accuracy: {} - Test Image Recall: {} - Test Image Precision: {}'.format(test_img_results[0], test_img_results[1], test_img_results[2], test_img_results[3]))

      pass

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

        pass


    # def compute_rmse(self, X_test, y_test, show=False):

    #     ## model is outputting a prediction of cancer class
    #     ## prediction is either right or wrong

    #     """
    #     compute rmse/measure to evalute model
    #     """
    #     pass


    def save_model(self):
        """
        Save the model into a .joblib
        """
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))


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
    # JOHN: get_data()
    # CAM: clean_data()
    df = pd.read_csv("../dataset/HAM10000_metadata.csv")
    X = df.drop(columns=['dx'])
    y = df['dx']
    t = Trainer(X, y)
    print("############  Preprocessing data   ############")
    t.preprocess()
    print("############  Training model   ############")
    t.train_predict()
    print("############  Evaluating model   ############")
    t.evaluate()
    # or score model etc





## Matt qs:
        ## should we write an evaluate function? model.evalute for cnns
        ## if we are writing evaluate function: y_test/pred = 7 column OHE matrix
        ## either: convert back to classes/0-6 numbers OR map one matrix onto another (TRUE/FALSE) and
        ## take number of rows containing FALSE / total number of rows
