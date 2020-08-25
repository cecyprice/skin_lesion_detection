import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from skin_lesion_detection.data import get_data, clean_df



def baseline_model(df, model='logistic regression', test_size=0.2):

    df = clean_df(df)

    X = df.drop(['lesion_id', 'image_id', 'dx', 'dx_type', 'sex', 'localization'], axis=1)
    y = df['dx']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    if model == 'logistic regression':
        log_model = LogisticRegression()
        log_model.fit(X_train, y_train)
        return log_model.score(X_test, y_test)
    if model == 'nearest neighbors':
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        return knn_model.score(X_test, y_test)



if __name__ == '__main__':
  print('implemented baseline model')
