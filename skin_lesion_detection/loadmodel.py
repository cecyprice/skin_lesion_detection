# load json and create model

from keras.models import model_from_json
from tensorflow import keras
from keras.optimizers import Adam

## Replace modelname with the name of model run you wish to load

name='Densenet_test'

json_file = open(f'{name}', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Replace weightsname with name used to save model weights in previous run

loaded_model.load_weights(f'{name}.h5')

opt = Adam(lr=1e-3, decay=1e-3 / 200)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print("--------------- MODEL LOADED ---------------------")

# Compile model and predict on test data

# prediction = loaded_model.predict(x=[self.X_met_train, self.X_im_train], verbose=0)
