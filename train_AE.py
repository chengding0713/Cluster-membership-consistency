""" construct the autoencoder model
"""
import keras
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Reshape
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras import backend as K
from keras.losses import binary_crossentropy
import numpy as np
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
import os
import pickle as pk

class ae():
    def __init__(self, verbose=True):
        input_dim = 2400
        filter_order = 20
        n_filters = 25
        feature_dim = 25
        self.verbose = verbose

        if input_dim % feature_dim is not 0:
            print('Feature dimensionality must be divisor of input dimensionality')

        # Input Layer
        self.input = keras.layers.Input(shape=(input_dim,))

        # Locally Connected Layer
        self.encoded = keras.layers.Reshape((input_dim, 1))(self.input)

        self.encoded = keras.layers.Conv1D(n_filters,filter_order,strides=input_dim//feature_dim,activation='relu',padding='same')(self.encoded)
        self.encoded = keras.layers.Reshape((input_dim//(input_dim//feature_dim),n_filters,1))(self.encoded)
        self.encoded = keras.layers.Conv2D(1, (1,n_filters),strides=(1,1),padding='valid')(self.encoded)

        # Convolutional Layer

        self.decoded = keras.layers.Conv2DTranspose(n_filters, (1,n_filters),strides=(1,1),padding='same')(self.encoded)
        #self.decoded = keras.layers.Reshape((feature_dim,1,1))(self.encoded)
        self.encoded = keras.layers.Reshape((feature_dim,))(self.encoded)

        self.decoded = keras.layers.Conv2DTranspose(1,(filter_order,1),strides=(input_dim//feature_dim,1),padding='same')(self.decoded)
        self.decoded = keras.layers.Reshape((input_dim,))(self.decoded)

        self.encoder = keras.Model(self.input,self.encoded)
        self.autoencoder = keras.Model(self.input,self.decoded)

        self.optimizer = keras.optimizers.Adam(lr=0.0001)
        self.autoencoder.compile(optimizer=self.optimizer, loss='mse')

        self.autoencoder.summary()


    def fit(self,inputs):
        self.autoencoder.fit(inputs,inputs,epochs=2,verbose=1,batch_size=500)

    def transform(self,inputs):
        if len(inputs.shape) is 1:
            inputs = inputs.reshape(1,-1)
        return self.encoder.predict(inputs)

    def test(self,inputs):
        if len(inputs.shape) is 1:
            inputs = inputs.reshape(1,-1)
        return self.autoencoder.predict(inputs)

    def save(self,filename):
        self.encoder.save(filename)




if __name__ == "__main__":
    batch_size = 512
    nb_classes = 2
    data_path = '/opt/localdata/storage/chengding_project_data/alarm_data_npy/'
    AF = np.float16(
        np.load('/opt/localdata/storage/stark_stuff/ppg_ecg_project/data/AF_v5/train_PPG_resampled2400.npy'))
    NSR = np.float16(
        np.load('/opt/localdata/storage/stark_stuff/ppg_ecg_project/data/NSR_v5/train_PPG_resampled2400.npy'))
    PVC = np.float16(
        np.load('/opt/localdata/storage/stark_stuff/ppg_ecg_project/data/PVC_v5/train_PPG_resampled2400.npy'))
    data = np.concatenate([AF, PVC, NSR])
    data = np.float16(data)

    path = data_path + 'AE_model/'
    if not os.path.exists(path):
        os.mkdir(path)

    checkpoint = ModelCheckpoint((path + '/' + 'resnet.{epoch:02d}-{val_acc:.2f}.hdf5'), verbose=1,
                                 monitor='val_loss', save_best_only=False, mode='auto')
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger(path + '/' + 'AE_PPG.csv')
    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    a1 = ae()

    a1.fit(data)
    x_train_ae = a1.transform(data)
    np.save('/opt/localdata/storage/chengding_project_data/alarm_data_npy/x_train_ae_2400.npy',x_train_ae)
