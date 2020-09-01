# load json and create model


## Replace modelname with the name you used to save the model in a previous run

model_file_name='modelname.json'

json_file = open('{}'.format(model_file_name), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Replace weightsname with name used to save model weights in previous run

model_weights_name='weightsname.h5'
loaded_model.load_weights('{}',format(model_weights_name))
print("Loaded model from disk")

# Compile model and predict on test data

opt = Adam(lr=1e-3, decay=1e-3 / 200)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
prediction = loaded_model.predict(x=[self.X_met_train, self.X_im_train], verbose=0)
