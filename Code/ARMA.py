from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

# Reading in data
df = pd.read_csv("Data/Shark_Slough.csv",index_col= "date", parse_dates = True)

# Selecting X.stn and Stage_cm
TS = df.loc[:, ["X.stn","Stage_cm"]]
TS.head()

# Selecting only station A13
P33 = TS[TS['X.stn'] == "P33"].loc[:,"Stage_cm"] # P33 has very low missingness, good for training rnn
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

train_size = int(len(P33) * 0.9) # Use 95% of data for training

train = P33.iloc[0:train_size]
test = P33.iloc[train_size:train_size + 31] 


f = pyplot.figure()
ax1 = f.add_subplot(121)
ax1.plot(P33.diff())
ax1.set_title("First Order Differencing")

ax2 = f.add_subplot(122)
ax2.set_title("ACF")
plot_pacf(P33, lags = 25, ax = ax2)
pyplot.show()

# Fit ARIMA

model = auto_arima(train)
model.fit(train)
model.summary()
arima_preds = model.predict(n_periods = 31)

plt.plot(test, color = 'b', label = True)
plt.plot(arima_preds, color = 'g', label = True)
plt.show()