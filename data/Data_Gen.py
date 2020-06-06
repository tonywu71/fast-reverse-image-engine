import numpy as np
import keras
import os

from data.data import *
# from data.data import load_image


class DataGenerator(keras.utils.Sequence):
    # CHANGE NEEDED: rename lista to sth more explicit (and not in portuguese please :) )
    # I think it is the list of classes, if that's it please rename it ASAP.
    '''Generates data for Keras'''
    def __init__(self, list_IDs, lista,db, batch_size=32, dim=(224,224), n_channels=3,
                 n_classes=20, shuffle=True, pre=lambda x: np.array(x)):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.lista = lista
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.db = db
        self.pre = pre

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):
        '''Generates data containing batch_size samples''' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            image_url = self.db.get_image(ID)
            image = load_image(image_url)
            X[i,] = self.pre(image)

            # Store class
            y[i] = self.db.get_label_labels(ID,self.lista)

        return X, y

class DataGenerator_training(DataGenerator):
    '''Special class for training that inherits from the DataGenerator class.'''

    def __data_generation(self, list_IDs_temp):
        '''Generates data containing batch_size samples'''

        def __init__(self, list_IDs_temp):
            self.training_directory = 'training_data'
            super().__init__(list_IDs_temp)
        

        # Initialisation
        X = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
        y = np.empty((self.batch_size,self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            ## HERE ARE THE CHANGES!!
            img_path = os.path.join(self.training_directory, str(ID))
            image = Image.open(img_path)
            X[i,] = self.pre(image)

            # Store class
            y[i] = self.db.get_label_labels(ID,self.lista)

        return X, y