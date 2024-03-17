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
                                batch_size = 40000) # Update network after 3 months of information, speeds up training

# Model Definition  
model = Sequential() # layers are added sequentially
model.add(LSTM(128, 
                activation = 'tanh', 
                input_shape = (n_input, n_features),
                return_sequences=True,
                kernel_regularizer=regularizers.L1(0.001),
                activity_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.3))
model.add(LSTM(64, 
                activation = 'tanh', 
                input_shape = (n_input, n_features),
                return_sequences=True,
                kernel_regularizer=regularizers.L1(0.001),
                activity_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.3))
model.add(LSTM(32, 
                activation = 'tanh', 
                input_shape = (n_input, n_features),
                return_sequences=False,
                kernel_regularizer=regularizers.L1(0.001),
                activity_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(optimizer = Adam(learning_rate=0.005), 
              loss = 'mse')
model.summary()

# Fitting model  
with tf.device('/device:GPU:0'): 
    model.fit(generator, epochs = 2000)

# Loop for training

# for j in range(10):
    
    # # Fitting model  
    # with tf.device('/device:GPU:0'): 
    #     model.fit(generator, epochs = 2000)

#     # Check when loss levels out
#     # loss_per_epoch = model.history.history["loss"]
#     # plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
#     # plt.show()

#     # True Out of Sample Predictions
#     duration = 31
#     test_predictions = []
#     test = P33_interp.iloc[train_size:train_size + duration] 
#     test = test.values.flatten()
#     test = test.reshape(-1,1)
#     first_eval_batch = scaled_train[-n_input:]
#     current_batch = first_eval_batch.reshape((1, n_input, n_features))
#     for i in range(duration):
#         current_pred = model.predict(current_batch)[0]
#         test_predictions.append(current_pred) 
#         current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
#     true_predictions = stage_transformer.inverse_transform(test_predictions)
#     plt.plot(test, color = 'b', label = True)
#     plt.plot(true_predictions, color = 'r', label = True)
#     plt.savefig(f"Model Diagnostics/model_{j}.png")
#     plt.clf()
        
        
# True Out of Sample Predictions
duration = 31
test_predictions = []
test = P33_interp.iloc[train_size:train_size + duration] 
test = test.values.flatten()
test = test.reshape(-1,1)
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))
for i in range(duration):
   current_pred = model.predict(current_batch)[0]
   test_predictions.append(current_pred) 
   current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = stage_transformer.inverse_transform(test_predictions)
plt.plot(test, color = 'b', label = True)
plt.plot(true_predictions, color = 'r', label = True)
plt.show()

metrics.mean_squared_error(test, true_predictions)
        # Best = 7.188 over 31 days

# Saving/Loading Best Model

# model.save("Models/Best_3_Layer_LSTM.keras")
# model = tf.keras.models.load_model('Models/Best_3_Layer_LSTM.keras')    