# ----------------------------------------------------------------------
# Libraries ------------------------------------------------------------
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
from keras.optimizers import Adam
import time
from keras import regularizers
from keras_tuner.tuners import BayesianOptimization

# Confirming GPU is being used
tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

# -----------------------------------------------------------------------
# Setting Up LSTM Architecture ------------------------------------------
# -----------------------------------------------------------------------

def build_model(hp):
    model = Sequential() # layers are added sequentially
    model.add(LSTM(hp.Int('layer_1_neurons', min_value = 8, max_value = 64), 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L1(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(hp.Choice('dropout_1', values = [0.01, 0.05, 0.1, 0.15])))
    model.add(LSTM(hp.Int('layer_2_neurons', min_value = 8, max_value = 64), 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L1(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(hp.Choice('dropout_2', values = [0.01, 0.05, 0.1, 0.15])))
    model.add(LSTM(hp.Int('layer_3_neurons', min_value = 8, max_value = 64), 
                    activation = 'tanh', 
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

# ------------------------------------------------------------------------
# EVER Data --------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
P33 = pd.read_csv("Data/P33.csv",index_col= "date", parse_dates = True)
P33 = P33[P33.index >= "1995-01-01"]

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(P33) * train_test_split) # Use 90% of data for training
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
n_input = 100
n_features = 1

# Define train and test
generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                length = n_input,
                                batch_size = 5000)

validation = TimeseriesGenerator(scaled_test, scaled_test, 
                                length = n_input,
                                batch_size = 1000)

# Initiate folder for saving searched models
LOG_DIR = f"{int(time.time())}" 

# Settings for trials
tuner = BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=50,
        executions_per_trial=1,
        directory = f"Models/EVER/{LOG_DIR}")

# Searching
tuner.search(
    x = generator,
    epochs = 100,
    validation_data = validation
    )

# Obtain best model from search
best_model_EVER = tuner.get_best_models()[0]

# Dimensions of model
best_model_EVER.summary()

# Getting layer specifications
for layer in best_model_EVER.layers:
    layer_config = layer.get_config()
    print(layer_config)

# Save best model
best_model_EVER.save("Models/Bayes_HT_EVER.keras")

# Load best model
best_model_EVER = tf.keras.models.load_model('Models/Bayes_HT_EVER.keras')    

# Training Best SSM Model

for j in range(12):
    
   # Fitting model  
   with tf.device('/device:GPU:0'): 
        best_model_EVER.fit(generator, epochs = 1000, validation_data = validation)

   duration = len(test)
   test_predictions = []
   first_eval_batch = scaled_train[-n_input:]
   current_batch = first_eval_batch.reshape((1, n_input, n_features))
   for i in range(duration):
      current_pred = best_model_EVER.predict(current_batch)[0]
      test_predictions.append(current_pred) 
      current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
   true_predictions = stage_transformer.inverse_transform(test_predictions)
   
   fig = plt.figure(figsize=(10,5))
   ax = fig.add_subplot(111)
   plt.plot(test, color = 'b', label = "EVER - P33")
   plt.plot(true_predictions, color = 'r', label = "LSTM Predictions")
   plt.legend()
   ax.set_ylabel("Response")
   ax.set_xlabel("Day's Past Training Data")
   ax.set_title("LSTM Predictions on Observed EVER Data")
   plt.savefig(f"Model Diagnostics/EVER_model_{j}.png")
   plt.clf()
   
# Check when loss levels out
loss_per_epoch = best_model_EVER.history.history["loss"]
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.show()

# ------------------------------------------------------------------------
# State Space Models -----------------------------------------------------
# ------------------------------------------------------------------------

# Load data
Sim_TS = pd.read_csv("Data/Simulated_SSM_TS.csv")
train = Sim_TS[:len(Sim_TS) - 120]
test = Sim_TS[len(Sim_TS) - 120:]

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
n_input = 100
n_features = 1

# Define train and test
generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                length = n_input,
                                batch_size = 5000)

validation = TimeseriesGenerator(scaled_test, scaled_test, 
                                length = n_input,
                                batch_size = 1000)

# Initiate folder for saving searched models
LOG_DIR = f"{int(time.time())}" 

# Settings for trials
tuner= BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=50,
        executions_per_trial=1,
        directory = f"Models/SSM/{LOG_DIR}")

# Searching
tuner.search(
    x = generator,
    epochs = 100,
    validation_data = validation
    )

# Obtain best model from search
best_model_SSM = tuner.get_best_models()[0]

# Dimensions of model
best_model_SSM.summary()

# Getting layer specifications
for layer in best_model_SSM.layers:
    layer_config = layer.get_config()
    print(layer_config)

# Save best model
best_model_SSM.save("Models/Bayes_HT_SSM.keras")

# Load best model
best_model_SSM = tf.keras.models.load_model('Models/Bayes_HT_SSM.keras')    

# Training Best SSM Model

for j in range(12):
    
   # Fitting model  
   with tf.device('/device:GPU:0'): 
        best_model_SSM.fit(generator, epochs = 1000, validation_data = validation)

   duration = len(test)
   test_predictions = []
   first_eval_batch = scaled_train[-n_input:]
   current_batch = first_eval_batch.reshape((1, n_input, n_features))
   for i in range(duration):
      current_pred = best_model_SSM.predict(current_batch)[0]
      test_predictions.append(current_pred) 
      current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
   true_predictions = stage_transformer.inverse_transform(test_predictions)
   
   fig = plt.figure(figsize=(10,5))
   ax = fig.add_subplot(111)
   plt.plot(test, color = 'b', label = "SSM")
   plt.plot(true_predictions, color = 'r', label = "LSTM Predictions")
   plt.legend()
   ax.set_ylabel("Response")
   ax.set_xlabel("Day's Past Training Data")
   ax.set_title("LSTM Predictions on Simulated SSM Data")
   plt.savefig(f"Model Diagnostics/SSM_model_{j}.png")
   plt.clf()
   
# Check when loss levels out
loss_per_epoch = best_model_SSM.history.history["loss"]
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.show()
   
# --------------------------------------------------------------
# ARIMA --------------------------------------------------------
# --------------------------------------------------------------

# Load data
Sim_TS = pd.read_csv("Data/Simulated_ARIMA_TS.csv")
train = Sim_TS[:len(Sim_TS) - 120]
test = Sim_TS[len(Sim_TS) - 120:]

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
n_input = 100
n_features = 1

# Define train and test
generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                length = n_input,
                                batch_size = 5000)

validation = TimeseriesGenerator(scaled_test, scaled_test, 
                                length = n_input,
                                batch_size = 1000)

# Initiate folder for saving searched models
LOG_DIR = f"{int(time.time())}" 

# Settings for trials
tuner= BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=50,
        executions_per_trial=1,
        directory = f"Models/ARIMA/{LOG_DIR}")

# Searching
tuner.search(
    x = generator,
    epochs = 100,
    validation_data = validation
    )

# Obtain best model from search
best_model_ARIMA = tuner.get_best_models()[0]

# Dimensions of model
best_model_ARIMA.summary()

# Getting layer specifications
for layer in best_model_ARIMA.layers:
    layer_config = layer.get_config()
    print(layer_config)

# Save best model
best_model_ARIMA.save("Models/Bayes_HT_ARIMA.keras")

# Load best model
best_model_ARIMA = tf.keras.models.load_model('Models/Bayes_HT_ARIMA.keras')    

# Training Best ARIMA Model

for j in range(12):
    
   # Fitting model  
   with tf.device('/device:GPU:0'): 
        best_model_ARIMA.fit(generator, epochs = 1000, validation_data = validation)

   duration = len(test)
   test_predictions = []
   first_eval_batch = scaled_train[-n_input:]
   current_batch = first_eval_batch.reshape((1, n_input, n_features))
   for i in range(duration):
      current_pred = best_model_ARIMA.predict(current_batch)[0]
      test_predictions.append(current_pred) 
      current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
   true_predictions = stage_transformer.inverse_transform(test_predictions)
   
   fig = plt.figure(figsize=(10,5))
   ax = fig.add_subplot(111)
   plt.plot(test, color = 'b', label = "ARIMA")
   plt.plot(true_predictions, color = 'r', label = "LSTM Predictions")
   plt.legend()
   ax.set_ylabel("Response")
   ax.set_xlabel("Day's Past Training Data")
   ax.set_title("LSTM Predictions on Simulated ARIMA Data")
   plt.savefig(f"Model Diagnostics/ARIMA_model_{j}.png")
   plt.clf()

# Plot training & validation loss values
plt.plot(best_model_ARIMA.history['loss'])
plt.plot(best_model_ARIMA.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()