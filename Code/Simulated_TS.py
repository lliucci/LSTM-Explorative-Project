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
from keras import callbacks
from tsbootstrap import MovingBlockBootstrap
from statsmodels.tsa.arima.model import ARIMA
from tsbootstrap import MovingBlockBootstrap

# Data Preprocessing --------------------------------------------------

# Sim_TS = pd.read_csv("Data/Simulated_SSM_TS.csv")
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

# Building Model -------------------------------------------

callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0005, patience=200)

# Define inputs  
n_input = 100
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                length = n_input,
                                batch_size = 5000)

model = Sequential() # layers are added sequentially
model.add(LSTM(32, 
                activation = 'tanh', 
                input_shape = (n_input, n_features),
                return_sequences=True,
                kernel_regularizer=regularizers.L2(0.001),
                activity_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.15))
model.add(LSTM(32, 
                activation = 'tanh', 
                input_shape = (n_input, n_features),
                return_sequences=True,
                kernel_regularizer=regularizers.L2(0.001),
                activity_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.15))
model.add(LSTM(32, 
                activation = 'tanh', 
                input_shape = (n_input, n_features),
                return_sequences=False,
                kernel_regularizer=regularizers.L2(0.001),
                activity_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(optimizer = Adam(learning_rate=0.001,
                            clipnorm = 1), 
            loss = 'mse')

gen_output = TimeseriesGenerator(scaled_test, scaled_test, 
                                length = n_input,
                                batch_size = 100)

# Training --------------------------------------------------

# Fitting model  
with tf.device('/device:GPU:0'): 
   model.fit(generator, epochs = 12000, validation_data = gen_output, callbacks = callback)
   
# True Out of Sample Predictions
duration = len(test)
test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))
for i in range(duration):
   current_pred = model.predict(current_batch)[0]
   test_predictions.append(current_pred) 
   current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = stage_transformer.inverse_transform(test_predictions)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test, color = 'b', label = "ARIMA")
plt.plot(true_predictions, color = 'r', label = "Predicted")
plt.legend()
ax.set_ylabel("Response")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Forecast of Simulated ARIMA Data")
plt.show()

# Loop for training ---------------------------------------------------------

for j in range(12):
    
   # Fitting model  
   with tf.device('/device:GPU:0'): 
        model.fit(generator, epochs = 1000, validation_data = gen_output)

   duration = len(test)
   test_predictions = []
   first_eval_batch = scaled_train[-n_input:]
   current_batch = first_eval_batch.reshape((1, n_input, n_features))
   for i in range(duration):
      current_pred = model.predict(current_batch)[0]
      test_predictions.append(current_pred) 
      current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
   true_predictions = stage_transformer.inverse_transform(test_predictions)
   
   fig = plt.figure(figsize=(10,5))
   ax = fig.add_subplot(111)
   plt.plot(test, color = 'b', label = "Predicted")
   plt.plot(true_predictions, color = 'r', label = "ARIMA")
   plt.legend()
   ax.set_ylabel("Response")
   ax.set_xlabel("Day's Past Training Data")
   ax.set_title("Comparison of Forecasts")
   plt.savefig(f"Model Diagnostics/ARIMA_model_{j}.png")
   plt.clf()

metrics.mean_squared_error(test, true_predictions)

# Prediction Intervals --------------------------------------------------------------

true_train = Sim_TS[:len(Sim_TS) - 120]
true_test = Sim_TS[len(Sim_TS) - 120:]

# Reshaping data sets from Panda Series to 1D Array
true_train = train.values.flatten()
true_train = true_train.reshape(-1,1)
true_test = test.values.flatten()
true_test = true_test.reshape(-1,1)

# Instantiate the bootstrap object
n_bootstraps = 1
block_length = 31
rng = 42
mbb = MovingBlockBootstrap(n_bootstraps=n_bootstraps, rng=rng, block_length=block_length)

stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(true_train)
true_scaled_train = stage_transformer.transform(true_train)


Boots = []


for j in range(1):
    
    # Generate bootstrapped samples
    return_indices = False
    bootstrapped_samples = mbb.bootstrap(
        Sim_TS, return_indices=return_indices)
        
    # Collect bootstrap samples
    X_bootstrapped = []
    for data in bootstrapped_samples:
        X_bootstrapped.append(data)

    X_bootstrapped = np.array(X_bootstrapped)

    TS_BS = pd.DataFrame(data=X_bootstrapped[0,:,0])

    # Splitting dataset for cross-validation
    train_size = int(len(TS_BS) * 0.9) # Use 90% of data for training
    train = TS_BS.iloc[0:train_size,0]
    test = TS_BS.iloc[train_size:,0]
    
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
    generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                    length = n_input,
                                    batch_size = 10000) # Update network after 3 months of information, speeds up training

    model = Sequential() # layers are added sequentially
    model.add(LSTM(32, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.15))
    model.add(LSTM(32, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.15))
    model.add(LSTM(32, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=False,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.05))
    model.add(Dense(1))
    model.compile(optimizer = Adam(learning_rate=0.001,
                                clipnorm = 1), 
                loss = 'mse')
    gen_output = TimeseriesGenerator(scaled_test, scaled_test, 
                                    length = n_input,
                                    batch_size = 1000)

    with tf.device('/device:GPU:0'): 
        model.fit(generator, epochs = 10, validation_data = gen_output, callbacks = callback)

    duration = len(true_test)
    test_predictions = []
    first_eval_batch = true_scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(duration):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred) 
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    true_predictions = stage_transformer.inverse_transform(test_predictions)
    Boots.append([z[0] for z in true_predictions])

Plotting = [
    [i[e] for i in Boots] #... take the eth element from ith array
    for e in range(len(Boots[0])) # for each e in 0:30...
    ]

sim_ts_df = pd.DataFrame(Plotting)
sim_ts_df.to_csv("Data/ARIMA_Pred_Intervals.csv")

# Load in ARIMA Intervals
Plotting = pd.read_csv("Data/ARIMA_Pred_Intervals.csv", index_col = 0)

# Plots bootstrapped predictions
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(true_test.values(), color = 'b', label = "Observed")
plt.plot(Plotting, alpha = 0.1, color = 'r')
plt.legend()
ax.set_ylabel("Response")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts")
plt.show()