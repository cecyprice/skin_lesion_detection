from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


def feature_encoding():
    ## function encodes gender and cancer diagnosis.
    ohe = OneHotEncoder(sparse = False)
    le = LabelEncoder()
    data['sex_encoded'] = le.fit_transform(data['sex'])
    feature_oh = ohe.fit_transform(data[['dx']])
    data['bkl'], data['nv'], data['df'], data['mel'], data['vasc'], data['bcc'], data['akiec'] = feature_oh.T
    return data


"""unfinished general encoder"""
# def general_feature_encoding2(v):
#     ## unfinished
#     ohe = OneHotEncoder(sparse = False)
#     feature_oh = ohe.fit_transform(data[[v]])
#     list = data[v].unique().tolist()
#     for l in list:
#         count = 0
#         while count <= len(list):
#             data['l'] = feature_oh[:,(count-1)]
#             count += 1
#     return data

