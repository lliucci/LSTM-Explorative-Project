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

# Reading in data
P33 = pd.read_csv("Data/P33.csv", index_col = 'date', parse_dates = True)
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

# Obtain structure of model chosen through hyperparameter tuning
best = tf.keras.models.load_model('Models/Best_HT_LSTM_20530.keras')    
best.summary()
best.get_config()

# Dataframe for saving predictions
Predictions = pd.DataFrame(index = range(31), columns=["Model_1", "Model_2", "Model_3", "Model_4", "Model_5"])

# Train 5 different models with same structure
for j in range(5):
    model = Sequential() # layers are added sequentially
    model.add(LSTM(17, 
                    activation = 'relu', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.15))
    model.add(LSTM(11, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.15))
    model.add(LSTM(37, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=False,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.05))
    model.add(Dense(1))
    model.compile(optimizer = Adam(learning_rate=0.0001,
                                clipnorm = 1), 
                loss = 'mse')
    gen_output = TimeseriesGenerator(scaled_test, scaled_test, 
                                    length = n_input,
                                    batch_size = 1000)
    for z in range(5):
        with tf.device('/device:GPU:0'): 
            model.fit(generator, epochs = 2000, validation_data = gen_output)

        # Predictions
        duration = 31
        test_predictions = []
        test = P33.iloc[train_size:train_size + duration] 
        test = test.values.flatten()
        test = test.reshape(-1,1)
        first_eval_batch = scaled_train[-n_input:]
        current_batch = first_eval_batch.reshape((1, n_input, n_features))
        for i in range(duration):
            current_pred = model.predict(current_batch)[0]
            test_predictions.append(current_pred) 
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        true_predictions = stage_transformer.inverse_transform(test_predictions)
        locals()[f'Model_{j}_Predictions_Gen_{z}'] = true_predictions
        locals()[f'Model_{j}_RMSE_Gen_{z}'] = metrics.mean_squared_error(test, true_predictions)

# Comparing models in generation 1
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test, color = 'b', label = "Observed")
plt.plot(Model_0_Predictions_Gen_0, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_0, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_0, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_0, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_0, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Depth (feet)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 1)")
plt.show()

# Comparing models in generation 2
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test, color = 'b', label = "Observed")
plt.plot(Model_0_Predictions_Gen_1, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_1, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_1, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_1, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_1, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Depth (feet)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 2)")
plt.show()

# Comparing models in generation 3
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test, color = 'b', label = "Observed")
plt.plot(Model_0_Predictions_Gen_2, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_2, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_2, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_2, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_2, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Depth (feet)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 3)")
plt.show()


# Comparing models in generation 4
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test, color = 'b', label = "Observed")
plt.plot(Model_0_Predictions_Gen_3, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_3, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_3, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_3, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_3, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Depth (feet)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 4)")
plt.show()


# Comparing models in generation 5
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test, color = 'b', label = "Observed")
plt.plot(Model_0_Predictions_Gen_4, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_4, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_4, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_4, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_4, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Depth (feet)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 5)")
plt.show()


# Comparison of RMSE over generations
x = [1, 2]
y1 = [Model_1_RMSE_Gen_0, Model_1_RMSE_Gen_1]
y2 = [Model_2_RMSE_Gen_0, Model_2_RMSE_Gen_1]
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.stem(x, y1)
ax.stem(x, y2)
plt.legend()
ax.set_ylabel("Root Mean Square Error")
ax.set_xlabel("Generations")
ax.set_title("Comparison of RMSE")
plt.show()