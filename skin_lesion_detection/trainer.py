from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler

import pandas as pd
import warnings

class Trainer(object):

    def __init__(self, X, y, **kwargs):

        self.pipeline = None
        self.kwargs = kwargs
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)
        if self.split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, random_state=1, test_size=0.3)


    def get_estimator(self):

        pass

    def set_pipeline(self):

        # Define feature engineering pipeline blocks
        pipe_multicat = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
        pipe_cont = make_pipeline(RobustScaler())
        pipe_binarycat = make_pipeline(LabelEncoder())
        # pipe_photo = make_pipeline(CUSTOMSCALERFORPIXELDATA())


        # Define default feature engineering blocs
        feateng_blocks = [
            ('multi_cat_feats', pipe_multicat, ['localization', 'dx_type']),
            ('cont_features', pipe_cont, ['age']),
            ('binary_cat_feats', pipe_binarycat, ['sex'])
            # ('photo_feats', pipe_photo, LISTOFPIXELCOLUMNS),
        ]

        features_encoder = ColumnTransformer(feateng_blocks, n_jobs=None, remainder="drop")

        self.pipeline = Pipeline(steps=[
            ('features', features_encoder)
            # ('model', self.get_estimator())
            ])


    def add_grid_search(self):
        """"
        Apply Gridsearch on self.params defined in get_estimator - using RegressionHyperModel?
        """
        pass


    #@simple_time_tracker
    def train(self, gridsearch=False):
        """
        Add time tracker - if we want?
        """
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)


    def evaluate(self):
        """
        evaluate performance using eg rmse
        """
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

    df = pd.read_csv("../dataset/HAM10000_metadata.csv")
    X = df.drop(columns=['dx'])
    y = df['dx']
    t = Trainer(X, y)
    print("############  Training model   ############")
    t.train()
    print(t.X_train.shape, t.y_train.shape)
