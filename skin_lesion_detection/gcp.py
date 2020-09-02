import json
import os
import joblib

from google.cloud import storage
from google.oauth2 import service_account
from termcolor import colored
from skin_lesion_detection.params import BUCKET_NAME, PROJECT_ID, MODEL_NAME, MODEL_VERSION

from keras.models import model_from_json
import h5py


def get_credentials():
    credentials_raw = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if '.json' in credentials_raw:
        credentials_raw = open(credentials_raw).read()
    creds_json = json.loads(credentials_raw)
    creds_gcp = service_account.Credentials.from_service_account_info(creds_json)
    return creds_gcp


def storage_upload(name, model_version=MODEL_VERSION, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_version,
        name)
    blob = client.blob(storage_location)
    blob.upload_from_filename(name)
    print(colored("=> model uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')


def download_model(name, extension, model_version=MODEL_VERSION, bucket=BUCKET_NAME, rm=True):
    creds = get_credentials()
    client = storage.Client(credentials=creds, project=PROJECT_ID).bucket(bucket)

    #storage_location = '/models/{}/versions/{}/{}'.format(
    #    MODEL_NAME,
    #    model_version,
    #    f"{name}.{extension}")
    #storage_location = '{}'.format(f"{name}.{extension}")
    print(storage_location)
    blob = client.blob(storage_location)
    blob.download_to_filename(f'{name}.{extension}')
    print(name)
    print(f"=> pipeline downloaded from storage")
    #model = joblib.load(name)
    #if rm:
    #    os.remove('model.joblib')
