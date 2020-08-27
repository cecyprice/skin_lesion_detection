import tensorflow as tf
from tf import keras
from tf.keras import Sequential
from tf.keras.applications.densenet import Densenet121
from tf.keras.applications import VGG16, ResNet50
from tf.keras.callbacks import EarlyStopping


class CNN_model():
    # Refactor function into the CNN_model class

def build_model(selection='vgg16'):
    '''
    Use VGG16, Resnet or Densenet as the base model
    Freeze pre-trained layers and add final layers to selected model
    '''
    #Specify input shape of the data
    input_shape = (75, 100, 3)

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

    x = model.layers.output
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(7, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=output)

    return model


def compile_model(model):
    '''
    Compile the model with adam optimizer, choosing accuracy as the metrics
    '''
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def fit_model(model):
    '''
    Fit the model using early stopping criteria
    '''
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                      validation_split=0.2,
                      epochs=100,
                      batch_size=16,
                      callbacks=[es])

    return history



if __name__ == "__main__":
    print('######### Building the model #########')
    model = build_model(model)
    print(model.summary())
    print('######### Compiling the model #########')
    model = compile_model(model)
    print('######### Fitting the model #########')
    model_hist = fit_model(model)
    print('######### Evaluating the model #########')
    print(f'Model accuracy: {model.evaluate(X_test, y_test)[1]}')

