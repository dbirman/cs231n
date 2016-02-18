import numpy as np

from assets.keras.keras.datasets import shapes_3d
from assets.keras.keras.preprocessing.image import ImageDataGenerator
from assets.keras.keras.models import Sequential
from assets.keras.keras.layers.core import Dense, Dropout, Activation, Flatten
from assets.keras.keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from assets.keras.keras.optimizers import SGD, RMSprop
from assets.keras.keras.utils import np_utils, generic_utils
from assets.keras.keras.regularizers import l2
import theano

#load dataset from gen_dataset
import cPickle as pickle
data = pickle.load(open('assets/data/lr_ds.data','rb'))

###

X_train = data['X_train']
X_train = np.expand_dims(X_train,axis=1)
Y_train = data['y_train']
X_val = data['X_val']
X_val = np.expand_dims(X_val,axis=1)
Y_val = data['y_val']
X_test = data['X_test']
X_test = np.expand_dims(X_test,axis=1)
Y_test = data['y_test']
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', Y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', Y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', Y_test.shape

###
#X_train = X_train[0:200,:,:,:,:]
#Y_train = Y_train[0:200]
#X_test = X_test[0:40,:,:,:,:]
#Y_test = Y_test[0:40]

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train-1, 2)
Y_test = np_utils.to_categorical(Y_test-1, 2)

###
__author__ = 'Dan Birman'



"""
    To classify/track 3D shapes, such as human hands (http://www.dbs.ifi.lmu.de/~yu_k/icml2010_3dcnn.pdf),
    we first need to find a distinct set of features. Specifically for 3D shapes, robust classification can be done using
    3D features.

    Features can be extracted by applying a 3D filters. We can auto learn these filters using 3D deep learning.

    This example trains a simple network for classifying 3D shapes (Spheres, and Cubes).

    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python shapes_3d_cnn.py

    CPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python shapes_3d_cnn.py

    For 4000 training samples and 1000 test samples.
    90% accuracy reached after 40 epochs, 37 seconds/epoch on GTX Titan
"""

# Data Generation parameters
#test_split = 0.2
#dataset_size = 100
#patch_size = 16

#(X_train, Y_train),(X_test, Y_test) = shapes_3d.load_data(test_split=test_split,
#                                                          dataset_size=dataset_size,
#                                                          patch_size=patch_size)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# CNN Training parameters
batch_size = 10
nb_classes = 2
nb_epoch = 50


# number of convolutional filters to use at each layer
nb_filters = [16, 32, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [7, 3, 3]

# Regularization
reg = 0

model = Sequential()
model.add(ZeroPadding3D((1,1,1),input_shape=(1,16,32,32)))
model.add(Convolution3D(nb_filters[0],nb_depth=1, nb_row=nb_conv[0], nb_col=nb_conv[0], border_mode='valid',
                         activation='relu', W_regularizer=l2(reg)))
#model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
model.add(Dropout(0.5))
model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(nb_filters[1],nb_depth=nb_conv[1], nb_row=nb_conv[1], nb_col=nb_conv[1], border_mode='valid',
                        activation='relu', W_regularizer=l2(reg)))
#model.add(MaxPooling3D(pool_size=(nb_pool[1], nb_pool[1], nb_pool[1])))
model.add(Dropout(0.5))
#model.add(ZeroPadding3D((1,1,1)))
#model.add(Convolution3D(nb_filters[2],nb_depth=nb_conv[2], nb_row=nb_conv[2], nb_col=nb_conv[2], border_mode='valid',
#                        activation='relu', W_regularizer=l2(reg)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(16, init='normal', activation='relu', W_regularizer=l2(reg)))
model.add(Dense(nb_classes, init='normal', W_regularizer=l2(reg)))
model.add(Activation('softmax'))

sgd = RMSprop(lr=.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])