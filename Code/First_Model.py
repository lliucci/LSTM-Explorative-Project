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
import random


# Confirming GPU is being used
import tensorflow as tf
tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Checking wd
os.getcwd()

# Reading in data
df = pd.read_csv("Data/Shark_Slough.csv",index_col= "date", parse_dates = True)

# Selecting X.stn and Depth
TS = df.loc[:, ["X.stn","Depth"]]
TS.head()

# Selecting only station A13
P33 = TS[TS['X.stn'] == "P33"].loc[:,"Depth"] # P33 has very low missingness, good for training rnn
P33.head

P33 = P33[P33.index >= "1995-01-01"]

print(P33.isnull().sum()) # 402 missing values present

#P33.plot(figsize=(12,6))

P33 = P33.interpolate(method = "linear") # Linear interpolation for missing values

print(P33.isnull().sum()) # 0 missing values present

#P33.plot(figsize=(12,6))
#plt.show()

# Length of time series
len(P33)

# Splitting dataset for cross-validation

train_size = int(len(P33) * 0.9) # Use 95% of data for training

train = P33.iloc[0:train_size]
test = P33.iloc[train_size:] 

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

scaled_train[:10]
scaled_test[:10]

# Define inputs  
n_input = 31
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                length = n_input,
                                batch_size = 10000) # Update network after 3 months of information, speeds up training

X,y = generator[0]

X.flatten() # first n_input days of information for training
y # n_input + 1 days value

# Model Definition  
model = Sequential() # layers are added sequentially
model.add(LSTM(128, 
               activation = 'tanh', 
               input_shape = (n_input, n_features)))
model.add(Dropout(rate = 0.3)) # 0.3
model.add(Dense(20))
model.add(Dense(1)) # final output layer
model.compile(optimizer = Adam(learning_rate=0.005), loss = 'mse')

model.summary()

# Fitting model  
with tf.device('/device:GPU:0'): 
    model.fit(generator, epochs = 2000)

# Check when loss levels out
loss_per_epoch = model.history.history["loss"]
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.show()

# Predictions
y_pred = model.predict(scaled_test)
true_pred = stage_transformer.inverse_transform(y_pred)

plt.plot(true_pred, label = True, color = "r")
plt.plot(test, label = True)
plt.show()

metrics.mean_squared_error(test, true_pred)

# Best: 0.0019

# Prediction for Another Station
Station = TS[TS['X.stn'] == "NP205"].loc[:,"Depth"] # P33 has very low missingness, good for training rnn
Station = Station[Station.index >= "2020-01-01"]
Station = Station.interpolate("linear")
Station = Station.values.flatten()
Station = Station.reshape(-1,1)

test_Station = stage_transformer.transform(Station)
y_pred = model.predict(test_DO2)
true_pred = stage_transformer.inverse_transform(y_pred)

plt.plot(true_pred, label = True, color = "r")
plt.plot(Station, label = True)
plt.show()

metrics.mean_squared_error(Station, true_pred)

# True Out of Sample Predictions
duration = 62

test_predictions = []

test = P33.iloc[train_size:train_size + duration] 
test = test.values.flatten()
test = test.reshape(-1,1)

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(duration):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
true_predictions = stage_transformer.inverse_transform(test_predictions)

plt.plot(test, color = 'b', label = True)
plt.plot(true_predictions, color = 'r', label = True)
plt.show()

# Saving/Loading Best Model

# model.save("Best.keras")
# model = tf.keras.models.load_model('Models/Best_3_Layer_LSTM.keras')