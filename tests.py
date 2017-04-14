import dataset
from utils import *
import numpy as np
import pyflux as pf
import pandas as pd
import matplotlib.pyplot as plt

sensorId = '55941031'

path = 'data/sensors.db'
db = dataset.connect('sqlite:///' + path, engine_kwargs={'connect_args': {'check_same_thread':False}})

vals = []
time_full = []
    
if db.tables.count(sensorId):  

    for el in db[sensorId].all():
        vals.append(float(el['data']))
        time_full.append(el['time'])
        
        
time_min = mean_minute_transform(vals, time_full)
        
x = np.asarray([to_local_time(ts, precision = "minutes") for ts in time_min.keys()])
y = np.asarray(time_min.values())


# truncating to values for April 13, 2017 
# x[3541] is '2017-04-13 06:17:00' 
# x[4108] is '2017-04-13 15:44:00'
x = x[3541:4108]
y = y[3541:4108]


# plotting the data converted to values averaged over minute-wide intervals
series = pd.Series(data = y, index = x)
series.plot(figsize=(15,10))
plt.show()


# using ARIMA model
model_arima = pf.ARIMA(data=np.asarray(y), ar=3, ma=1, target=np.asarray(x))

# training via maximum likelihood estimation
trained_arima = model_arima.fit("MLE")

# summary report on the trained model
print(trained_arima.summary())

# latent variable plot
model_arima.plot_z(figsize=(15,10))

# show goodness of fit
model_arima.plot_fit(figsize=(15,10))

# compare predictions for the last 10 steps / minutes, and compare it to the actual data
model_arima.plot_predict_is(h=10, past_values = 10, figsize=(15,10))

# predict the next 5 data points
model_arima.predict(h=5)


# model_llev = pf.LLEV(data=np.asarray(y),target=np.asarray(x))
# trained_llev = model.fit()
# print(trained_llev.summary())
# 
# model_llev.plot_fit(figsize=(15,10))
# 
# plt.figure(figsize=(15,10))
# for i in range(10):
#     plt.plot(model_llev.index, model_llev.simulation_smoother(
#             model_llev.latent_variables.get_z_values())[0][0:len(model_llev.index)])
# plt.show()
# 
# model_llev.plot_predict_is(h=10, figsize=(15,10))
# 
# model_llev.plot_predict(h=5, past_values=10, figsize=(15,10))
# 
# model_llev.predict(h=5)