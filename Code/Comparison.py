from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import RobustScaler
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
train_size = int(len(P33) * 0.9) # Use 95% of data for training
train = P33.iloc[0:train_size]
train = train.values.flatten()
train = train.reshape(-1,1)
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(train)
scaled_train = stage_transformer.transform(train)
n_input = 31
n_features = 1

# Fit ARIMA

model_arima = auto_arima(train)
model_arima.fit(train)
model_arima.summary()
arima_preds = model_arima.predict(n_periods = 31)
forecast = pd.DataFrame(arima_preds, index = P33.index[train_size:train_size + 31], columns=['Prediction'])

# Fit HW

HW = ExponentialSmoothing(train, trend = 'add', seasonal = 'add', seasonal_periods = 365).fit()
HW_preds = HW.forecast(31)

# Load LSTM

model = tf.keras.models.load_model('Models/Best_HT_LSTM_20530.keras')
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

metrics.mean_squared_error(test, true_predictions) # LSTM MSE
metrics.mean_squared_error(arima_preds, test) # ARIMA MSE

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test, color = 'b', label = "Observed")
plt.plot(true_predictions, color = 'r', label = "LSTM")
plt.plot(arima_preds, color = 'g', label = "ARIMA")
plt.plot(HW_preds, color = 'y', label = "Holt-Winters")
plt.legend()
ax.set_ylabel("Depth (feet)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts")
plt.show()