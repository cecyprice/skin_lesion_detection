from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler

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
        model = self.mixed_model #from mixed_model.py
        model.fit(x=[self.X_met_train, self.X_im_train], y=self.y_train,
        validation_split=0.3,
        epochs=200, batch_size=8)
        #Add Early Stopping??
        self.y_preds = model.predict([self.X_met_test, self.X_im_test])


    def evaluate(self):
        """
        evaluate performance using eg rmse
        """
        # Something like this...
            # diff = self.y_preds.flatten() - self.y_test
            # percentDiff = (diff / self.y_test) * 100
            # absPercentDiff = np.abs(percentDiff)
        pass



    def compute_rmse(self, X_test, y_test, show=False):
        """
        compute rmse/measure to evalute model
        """
        pass


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

