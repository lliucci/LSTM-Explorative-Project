from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from keras.optimizers import Adam
import time
from sklearn import metrics
from keras import regularizers


# Reading in data
df = pd.read_csv("Data/Shark_Slough.csv",index_col= "date", parse_dates = True)
# Selecting X.stn and Stage_cm
TS = df.loc[:, ["X.stn","Stage_cm"]]
# Selecting only station A13
P33 = TS[TS['X.stn'] == "P33"].loc[:,"Stage_cm"] # P33 has very low missingness, good for training rnn
P33 = P33[P33.index >= "1995-01-01"] # Start date for analysis
P33 = P33.interpolate(method = "linear") # Linear interpolation for missing values
# Length of time series
len(P33)
train_size = int(len(P33) * 0.9) # Use 95% of data for training
train = P33.iloc[0:train_size]
train = train.values.flatten()
train = train.reshape(-1,1)
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(train)

# Fit ARIMA

model = auto_arima(train)
model.fit(train)
model.summary()
arima_preds = model.predict(n_periods = 31)
forecast = pd.DataFrame(arima_preds,index = P33.index,columns=['Prediction'])

# Load LSTM

model = tf.keras.models.load_model('Best.keras')
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

plt.plot(test, color = 'b', label = True)
plt.plot(true_predictions, color = 'r', label = True)
plt.plot(arima_preds, color = 'g', label = True)
plt.show()