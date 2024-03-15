# Deep Learning Libraries

from functools import partial
# import keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, GlobalMaxPool1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Activation, concatenate, SpatialDropout1D, TimeDistributed, Layer, AlphaDropout, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import keras_tuner as kt

import numpy as np
import pandas as pd
# from keras import backend as K
from sklearn.model_selection import GroupShuffleSplit
from functools import partial
# from keras.callbacks import *
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score

import sklearn
from sklearn.metrics import confusion_matrix

# General Libraries
from scipy.io import loadmat, savemat
from scipy.fft import fft, fftfreq, ifft
import h5py
import os

## Get Data
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')

folderpath = '/data/users3/cellis42/Spectral_Explainability/PreTraining/Data/'
filepath = [folderpath + 'segmented_hc1_data_like_sleep.npy',
            folderpath + 'segmented_hc2_data_like_sleep.npy',
            folderpath + 'segmented_mdd1_data_like_sleep.npy',
            folderpath + 'segmented_mdd2_data_like_sleep.npy']

for i in np.arange(4):

    f = np.load(filepath[i],allow_pickle=True).item()
    
    if i == 0:
        data = f['data']
        labels = f['label']
        groups = f['subject']
    else:
        data = np.concatenate((data,f['data']),axis=0)
        labels = np.concatenate((labels,f['label']),axis=0)
        groups = np.concatenate((groups,f['subject']),axis=0)
        channels = f['channels']
                
channels2 = []
for i in range(19):
    channels2.append(channels[i].strip('EEG ').strip('-L'))

channels = channels2
channels2 = []

data = np.swapaxes(data,1,2)
data = np.expand_dims(data,axis=3)

## Define Base Model

class MyHyperModel(kt.HyperModel):
    
    def build(self,hp):

        n_timesteps = 3000
        n_features = 19

        convLayer2D = partial(Conv2D,activation='elu',kernel_initializer='he_normal',padding='valid',
                        kernel_constraint=max_norm(max_value = 1))
    
        convLayer1D = partial(Conv1D,activation='elu',kernel_initializer='he_normal',padding='valid',
                            kernel_constraint=max_norm(max_value = 1))

        inputs = Input(shape=(n_timesteps, n_features))

        x = tf.transpose(inputs,perm=[0,2,1])

        stride_kernel_size_0 = 50
        x = tf.keras.layers.Reshape((int(n_timesteps*n_features),1))(x)
        x = convLayer1D(filters = hp.Int("conv_units_0", min_value=15, max_value=30, step=5),
                        kernel_size = stride_kernel_size_0, 
                        strides = stride_kernel_size_0,
                        data_format='channels_last')(x)

        x = tf.expand_dims(x,axis=3)
        x_shape = x.shape
        x = tf.split(x,19,1)
        x = tf.keras.layers.Concatenate(axis=3)(x)
        x = tf.transpose(x,perm=[0,1,3,2])
        x = BatchNormalization()(x)
        
        stride_size_1 = hp.Choice("stride_size_1",[5,10])
        kernel_size_1 = hp.Choice("kernel_size_1",[5,10])
        x = convLayer2D(filters=1,
                        kernel_size= (kernel_size_1,1), 
                        strides=(stride_size_1,1),
                        data_format='channels_last')(x)
        x = tf.squeeze(x,axis=3)
        x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Choice("dropout_0",[0.0,0.1,0.2,0.3]))(x)
        outputs = Dense(2,activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        learning_rate = hp.Choice("lr",[1e-5,5e-5,7e-5,1e-4,5e-4,7e-4,1e-3])#hp.Float("lr", min_value=1e-5, max_value=1e-3, sampling="log")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                         metrics = ['acc'])

        return model

batch_norm_layer_idx = [2, 5, 8, 11]

## Train Model 2

class CVTuner(kt.Hyperband):#kt.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, groups, batch_size=128, epochs=10):
        val_accuracy = []
        gss = GroupShuffleSplit(n_splits = 10, train_size = 0.9, random_state=3) #
        for tv_idx, test_idx in gss.split(x, y, groups):
            X_train_val, X_test = x[tv_idx,...], x[test_idx,...]
            y_train_val, y_test = y[tv_idx], y[test_idx]

            group = groups[tv_idx]
            gss_train_val = GroupShuffleSplit(n_splits = 1, train_size = 0.78, random_state = 3) # random_state = 11
            for train_idx, val_idx in gss_train_val.split(X_train_val, y_train_val, group):

                X_train = X_train_val[train_idx,...]
                y_train = y_train_val[train_idx]

                X_val = X_train_val[val_idx,...]
                y_val = y_train_val[val_idx]
                
                print(np.shape(X_train))
                
                # Create Weights for Model Classes
                values, counts = np.unique(y_train, return_counts=True)

                weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = np.squeeze(y_train))
                class_weights = dict(zip(values, weights))
                
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_acc",patience=20)]

                model = self.hypermodel.build(trial.hyperparameters)
                model.fit(X_train, tf.keras.utils.to_categorical(y_train), validation_data=(X_val, tf.keras.utils.to_categorical(y_val)),
                          shuffle=True, verbose=0, batch_size=batch_size, epochs=epochs, class_weight=class_weights,callbacks=callbacks)
                val_accuracy.append(model.evaluate(X_val, tf.keras.utils.to_categorical(y_val))[1])
        
        print(trial.trial_id)
        val_accuracy = np.mean(val_accuracy)
        
        return val_accuracy
    # self.oracle.update_trial(trial.trial_id, {'val_accuracy': np.mean(val_accuracy)})
    # self.save_model(trial_id=trial.trial_id, model=model)

tuner = CVTuner(
    hypermodel=MyHyperModel(),
    objective = kt.Objective("val_accuracy", direction="max"),
    executions_per_trial=1, # number of times each trial is initialized (b/c different initializations get different results)
    max_epochs=200, #40
    overwrite=True,
    seed=0,
    directory="/data/users3/cellis42/Spectral_Explainability/InterpretableConv",
    project_name="Keras_tuner_logs_M3_XAIWC")

tuner.search(data, labels, groups, epochs=75)

# Print Best HyperParameters
best_hps= tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps) 
print(best_hps.values)
print(model.summary())

################################################################################################################
# Train Optimal Model 2

tf.random.set_seed(41) # best is seed 42, v7

testing_metrics = []; validation_metrics = [];

n_timesteps = 3000
n_features = 19

i = 0


gss = GroupShuffleSplit(n_splits = 50, train_size = 0.9, random_state = 3) # 11

for tv_idx, test_idx in gss.split(data, labels, groups):
    
    print(i)
    # if i == 2:
    # tf.keras.backend.clear_session()

    X_train_val = data[tv_idx,...]
    y_train_val = labels[tv_idx]

    X_test = data[test_idx,...]
    y_test = labels[test_idx]

    group = groups[tv_idx]
    gss_train_val = GroupShuffleSplit(n_splits = 1, train_size = 0.78, random_state = 3) #random_state = 11
    for train_idx, val_idx in gss_train_val.split(X_train_val, y_train_val, group):
        
        X_train = X_train_val[train_idx,...]
        y_train = y_train_val[train_idx]
        
        X_val = X_train_val[val_idx,...]
        y_val = y_train_val[val_idx]
        
        # Build the model with the optimal hyperparameters
        # train the model.
        best_hps= tuner.get_best_hyperparameters(1)[0]
        model = tuner.hypermodel.build(best_hps)        
        
        file_path = "/data/users3/cellis42/Spectral_Explainability/InterpretableConv/model_M3_XAIWC_kt_fold0"+str(i)+".hdf5"
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_acc",patience=20)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

        # Create Weights for Model Classes
        values, counts = np.unique(y_train, return_counts=True)

        weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = np.squeeze(y_train))
        class_weights = dict(zip(values, weights))
        
        history = model.fit(X_train, tf.keras.utils.to_categorical(y_train), epochs= 200, batch_size = 128, validation_data=(X_val, tf.keras.utils.to_categorical(y_val)), 
                            shuffle=True, verbose = 0, callbacks=[checkpoint,early_stopping],class_weight=class_weights)
            
        model.load_weights(file_path)

        preds = np.argmax(model.predict(X_test, batch_size=128),axis=1)

        testing_metrics.append([accuracy_score(y_test, preds),recall_score(y_test, preds, pos_label=1),recall_score(y_test, preds, pos_label=0),balanced_accuracy_score(y_test, preds)])
        
        preds_val = np.argmax(model.predict(X_val, batch_size=128),axis=1)

        validation_metrics.append([accuracy_score(y_val, preds_val),recall_score(y_val, preds_val, pos_label=1),recall_score(y_val, preds_val, pos_label=0),balanced_accuracy_score(y_val, preds_val)])
    
    i += 1
    

print("Validation Set Metrics")
validation_metrics = np.array(validation_metrics)
print(validation_metrics)
print(pd.DataFrame(data=[validation_metrics.mean(axis=0), validation_metrics.std(axis=0)], index=['mean','std'], columns=['acc','sens','spec','bacc']))

print("Test Set Metrics")
testing_metrics = np.array(testing_metrics)
print(testing_metrics)
print(pd.DataFrame(data=[testing_metrics.mean(axis=0), testing_metrics.std(axis=0)], index=['mean','std'], columns=['acc','sens','spec','bacc']))
    
results_filename = "/data/users3/cellis42/Spectral_Explainability/InterpretableConv/Performance_M3_XAIWC_kt.mat"
savemat(results_filename,{"validation_metrics":validation_metrics,"testing_metrics":testing_metrics})