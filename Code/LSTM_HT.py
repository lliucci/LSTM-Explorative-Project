import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
from keras.optimizers import Adam
import time
from sklearn import metrics
from keras import regularizers
import math
import keras
from tensorflow.keras.models import save_model
from tensorflow.keras.models import model_from_json
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from pickle import dump,load
import warnings
import keras_tuner


# Confirming GPU is being used
import tensorflow as tf
tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Checking wd
os.getcwd()

# Reading in data
P33 = pd.read_csv("Data/P33.csv",index_col= "date", parse_dates = True)
# Date filtering
P33 = P33[P33.index >= "1995-01-01"]
# Length of time series
len(P33)

# Splitting dataset for cross-validation
train_size = int(len(P33) * 0.9) # Use 90% of data for training
train = P33.iloc[0:train_size]
test = P33.iloc[train_size:len(P33)] 

# Reshaping data sets from Panda Series to 1D Array
train = train.values.flatten()
train = train.reshape(-1,1)
test = test.values.flatten()
test = test.reshape(-1,1)

# Scaling time-series train and test values values
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(train)
scaled_train = stage_transformer.transform(train)
scaled_test = stage_transformer.transform(test)

# Define inputs  
n_input = 31
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                length = n_input,
                                batch_size = 10000) # Update network after 3 months of information, speeds up training

validation = TimeseriesGenerator(scaled_test, scaled_test, 
                                length = n_input,
                                batch_size = 10000)

def build_model(hp):
    model = Sequential() # layers are added sequentially
    model.add(LSTM(hp.Int('layer_1_neurons', min_value = 8, max_value = 64), 
                    activation = hp.Choice('layer_1_activation', values = ['relu', 'tanh']), 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L1(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(hp.Choice('dropout_1', values = [0.01, 0.05, 0.1, 0.15])))
    model.add(LSTM(hp.Int('layer_2_neurons', min_value = 8, max_value = 64), 
                    activation = hp.Choice('layer_2_activation', values = ['relu', 'tanh']), 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L1(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(hp.Choice('dropout_2', values = [0.01, 0.05, 0.1, 0.15])))
    model.add(LSTM(hp.Int('layer_3_neurons', min_value = 8, max_value = 64), 
                    activation = hp.Choice('layer_3_activation', values = ['relu', 'tanh']), 
                    input_shape = (n_input, n_features),
                    return_sequences=False,
                    kernel_regularizer=regularizers.L1(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(hp.Choice('dropout_3', values = [0.01, 0.05, 0.1, 0.15])))
    model.add(Dense(1))
    model.compile(optimizer = Adam(learning_rate=0.001,
                                   clipnorm = 1,
                                   clipvalue = 0.5), 
                loss = 'mse')
    return(model)

LOG_DIR = f"{int(time.time())}" 

tuner= RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=50,
        executions_per_trial=1,
        directory = LOG_DIR
        )


tuner.search(
    x = generator,
    epochs = 30,
    validation_data = validation
    )

# Obtain best model from random search
best_model = tuner.get_best_models()[0]

# Train best model for an additional 100 epochs
best_model.fit(generator, epochs = 500, validation_data = validation)

# True Out of Sample Predictions
duration = 31
test_predictions = []
test = P33.iloc[train_size:train_size + duration] 
test = test.values.flatten()
test = test.reshape(-1,1)
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))
for i in range(duration):
    current_pred = best_model.predict(current_batch)[0]
    test_predictions.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = stage_transformer.inverse_transform(test_predictions)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test, color = 'b', label = 'Observed')
plt.plot(true_predictions, color = 'r', label = 'Predicted')
plt.legend()
ax.set_ylabel("Depth (feet)")
ax.set_xlabel("Day's Into The Future")
ax.set_title("Comparison of Forecasts")
plt.show()

metrics.mean_squared_error(test, true_predictions)

# Loop for training

for j in range(10):
    
    # Fitting model  
    with tf.device('/device:GPU:0'): 
        best_model.fit(generator, epochs = 2000)

    # Check when loss levels out
    # loss_per_epoch = model.history.history["loss"]
    # plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
    # plt.show()

    # True Out of Sample Predictions
    duration = 31
    test_predictions = []
    test = P33.iloc[train_size:train_size + duration] 
    test = test.values.flatten()
    test = test.reshape(-1,1)
    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(duration):
        current_pred = best_model.predict(current_batch)[0]
        test_predictions.append(current_pred) 
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    true_predictions = stage_transformer.inverse_transform(test_predictions)
    plt.plot(test, color = 'b', label = True)
    plt.plot(true_predictions, color = 'r', label = True)
    plt.savefig(f"Model Diagnostics/model_{j}.png")
    plt.clf()
        
# best_model.save("Models/Best_HT_LSTM_20530.keras")