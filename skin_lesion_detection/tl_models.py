import tensorflow.keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, concatenate, Dropout, Activation, MaxPooling2D, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications.resnet import ResNet50


class TLModels():
    # Refactor function into the CNN_model class

    def tl_create_mlp(self, input_dim):
        """
        Create Multi-Layer Perceptron as left_hand fork of mixed neural network for numeric and categorical explanatory variables
        """
        model = Sequential()
        model.add(Dense(8, input_dim=input_dim, activation="relu"))
        model.add(Dense(4, activation="relu"))
        return model


    def tl_create_cnn(self, input_shape, selection='vgg16', filters=(16, 32, 64)):
        '''
        Use VGG16, Resnet or Densenet as the base model
        Freeze pre-trained layers and add final layers to selected model
        '''
        # Implement VGG16 model
        if selection == 'vgg16':
            model = VGG16(weights='imagenet',
                          input_shape=input_shape,
                          include_top=False)

        # Implement ResNet model
        if selection == 'resnet':
            model = ResNet50(weights='imagenet',
                             input_shape=input_shape,
                             include_top=False,
                             classes=7)

        # Implement DenseNet model
        if selection == 'densenet':
            model = DenseNet121(weights='imagenet',
                                input_shape=input_shape,
                                include_top=False,
                                classes=7)

        # Make pre-trained layers non iterable and add final layers
        for layer in model.layers:
            layer.iterable = False
            layer.trainable = False

        inp = Input(shape=input_shape)
        base_output = model(inp)
        x = Flatten()(base_output)
        x = Dense(64, activation='relu')(x)
        x = Dense(4, activation='relu')(x)

        # construct the CNN and return model
        model = Model(inp, x)
        return model


    def tl_merge_compile_models(self, input_dim, input_shape, selection='vgg16', filters=(16, 32, 64), num_labels=7):
        """
        Join forks of network to combine models for all data types
        """
        # create the MLP and CNN models
        mlp = self.tl_create_mlp(input_dim)
        cnn = self.tl_create_cnn(input_shape)

        # create the input to our final set of layers as the output of both the MLP and CNN
        combinedInput = concatenate([mlp.output, cnn.output])

        # add final FC layer head with 2 dense layers with final layer as the multi-classifier head
        x = Dense(4, activation="relu")(combinedInput)
        x = Dense(num_labels, activation="softmax")(x)

        # yield final model integrating categorical/numerical data and images into single diagnostic prediction
        model = Model(inputs=[mlp.input, cnn.input], outputs=x)

        # compile the model using BCE as loss
        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss="categorical_crossentropy",
          optimizer=opt,
          metrics=['accuracy'])

        #NB have removed  'precision', 'f1'
        return model


    # def fit_model(model):
    #     '''
    #     Fit the model using early stopping criteria
    #     '''
    #     es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True)

    #     history = model.fit(X_train, y_train,
    #                       validation_split=0.2,
    #                       epochs=100,
    #                       batch_size=16,
    #                       callbacks=[es])
        # return history



# if __name__ == "__main__":
#     print('######### Building the model #########')
#     model = build_model(model)
#     print(model.summary())
#     print('######### Compiling the model #########')
#     model = compile_model(model)
#     print('######### Fitting the model #########')
#     model_hist = fit_model(model)
#     print('######### Evaluating the model #########')
#     print(f'Model accuracy: {model.evaluate(X_test, y_test)[1]}')

