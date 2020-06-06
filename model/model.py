## Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import io

# Tensorflow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Hyperparameters
from tensorboard.plugins.hparams import api as hp


# ------------------------------------------------------------------------------------------

assert(int(tf.__version__[0]) >= 2), "You are using TensorFlow 1, please update before proceeding!"

def get_data(data_path, img_size, batch_size, validation_split = 0.2, **kwargs):
    '''Given a directory containing all the images stored in folders named with the corresponding label,
    this function returns two data generators object.
    
    Inputs:
        - data_path = string
        - img_size = 2-tuple of the images' dimensions
        - batch_size = size of the batch (please input power of 2 for more efficiency)
        - validation_split = float between 0 and 1 that gives the percentage allocated to the validation set
        - **kwargs = optional dictionary which contains arguments for data augmentation
        
    Outputs:
        - ImageDataGenerator object
    '''
    
    datagen = ImageDataGenerator(rescale=1./255,
                                 preprocessing_function=preprocess_input,
                                 validation_split=validation_split,
                                 **kwargs)
    
    train_generator = datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset = 'training')

    validation_generator = datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset = 'validation')
    
    return train_generator, validation_generator


def show_img(data_gen, batch_idx, idx):
    ''''Display the prompted image with MatPlotLib.
    
    Inputs:
        - data_gen = a generator returned by get_data for instance
        - imgbatch_idx = the index of the batch
        - idx = the index of the image inside the selected batch
        
    Outputs:
        - an ImageDataGenerator object
    '''
    
    
    plt.imshow(data_gen[batch_idx][0][idx])
    return 


def show_label(data_gen, batch_idx, idx):
    ''''Returns the label of the prompted image.
    
    Inputs:
        - data_gen = a generator returned by get_data for instance
        - imgbatch_idx = the index of the batch
        - idx = the index of the image inside the selected batch
        
    Outputs:
        - a string
    '''
    
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    
    return labels[np.argmax(data_gen[batch_idx][1][idx])]


def get_batch_size(image_generator):
    '''
    Returns the number of batches of a set contained in an ImageGenerator object.
    
    Input:
        - image_generator = ImageGenerator object
    Output:
        - len(train_generator) = number of batches
    '''
    return len(train_generator)


def generate_model(hparams):
    '''
    A function that generates the model based on the hyperparameters that are given.
    The model is not yet compiled, neither with the weights

    Inputs:
        - hparams: a dictionnary containning, at least, the following hyperparameters (in parenthesis is the hyperparameter key)
          - 'HP_DROPOUT': value for dropout'
          - 'HP_NUM_UNITS_RELU': number of units in the second to last (Dense) layer
          - 'HP_NB_FROZEN_LAYERS': gives how many layers we'd like to freeze
          - 'HP_LOSS': choice of the loss function

    Output:
            - a model compiled but not trained, with the given hyperparamaters

    '''
    
    # Building the model step by step
    model = Sequential()
    model.add(ResNet50(include_top = False, weights = 'imagenet'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(Dense(units=hparams['HP_NUM_UNITS_RELU'], activation='relu'))
    model.add(Dense(102,activation='softmax'))
    
    model.compile(optimizer='adam', loss=hparams['HP_LOSS'], metrics=['accuracy'])
    
    # Freezing some layers depending of the corresponding hyperparameter  
    for i in range(hparams['HP_NB_FROZEN_LAYERS']):
            model.layers[0].layers[-(i+1)].trainable = False
    
    return model


def fit_model(model, train_generator, epochs):
    '''
    This function trains the model given as an input as well as logging some useful data in Tensorboard.
    SIDE-EFFECT as model is modified on place.
    
    Inputs:
        - model: a compiled but not trained model
        - train_generator: an ImageGenerator objects
        - epochs: number of epochs

    Output:
      - a History object
    '''
    
    # Tensorflow configuration
    logdir_scalars = 'logs/scalars/'
    
    # Creating callbacks
    tensorboard_callbacks = keras.callbacks.TensorBoard(log_dir=logdir_scalars)

    # Fitting the model
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=get_batch_size(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps= get_batch_size(validation_generator),
        verbose=1,
        shuffle=True,
        callbacks=[tensorboard_callbacks]
    )

    return history


def load_weights(model, weights_filepath):
    '''
    A function that loads weights to the model.
    The function raise an exception if the file is not a .h5

    Inputs:
      - model: a generated keras model
      - weights_filepath: the filepath to weights data, must be a .h5 file

    Output:
      - None
    '''
    
    assert os.isfile(weights_filepath), 'Weights file not found 👎🏻'
    print('Weights file found 🤙🏻')
    
    model.load_weights('weigths_filepath')
    return



