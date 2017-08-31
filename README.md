# SeismicServer

SeismicServer Statistical Prediction App receives a stream of data in the following format: <br/>
*sensorId=55941031&decimal=2.0&timestamp=1490710679* <br/>

The received data is visualized as a time-series chart and stored for further statistical predictions: one-step-ahead or 5-steps-ahead predictions. <br/>

*Main interface with a chart and prediction functionality:* http://127.0.0.1/ <br/>
*Example input request [insert new value]:* http://127.0.0.1/insert/sensorId=55941031&decimal=2.0&timestamp=1490710679 <br/>
*Example output request [return last value]:* http://127.0.0.1/last <br/>
*Example one-step-ahead prediction request:* http://127.0.0.1/predict_next <br/>
*Example 5-steps-ahead prediction request:* http://127.0.0.1/predict_next_5 <br/>
*Example accuracy test request [trains and tests incrementally over the last 10% of the data, may take a lot of time to complete]:* http://127.0.0.1/test
 <br/>
 <br/>

![SeismicServer Statistical Prediction App, beta version](/data/seismicserver_beta.png?raw=true "SeismicServer beta")

## Additional features

Each consecutive prediction provides an approximate likelihood estimate (in 0-1 range). <br/>

The likelihood estimate, in addition to the magnitude & relative-change for the next prediction, takes into account recent model's performance based on autonomous testing which occurs roughly every 30 minutes (depending on frequency of pushes from the sensor). <br/>

Autonomous testing occurs in a separate thread, the resulting accuracy snapshots are stored in separate .csv files for each sensor <br/>

The model's parameters are fine-tuned (including seasonal component of SARIMAX model) for two sensors: *55941031* and *2225280* <br/>

Accuracy snapshots (median accuracy over median accuracy for two 20 minute-wide sliding windows over last 50 minutes) can be found in *data/current_accuracy_%sensor_name%* <br/>

## Depdendencies

For Ubuntu 16.04 with Python 2.7.13 with SciPy stack (or Anaconda2):

```shell

pip install web.py

pip install plotly

sudo apt-get install xvfb libav-tools xorg-dev libsdl2-dev swig cmake

pip install statsmodels
```
For deploying on a Free-tier micro.t2 machine with 1GB RAM, enabling swap file might be needed (to install such dependencies as scipy, etc., if not using Anaconda):
```shell
sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=1024
sudo /sbin/mkswap /var/swap.1
sudo /sbin/swapon /var/swap.1
```

To deploy it on the default [80] port:

```shell
sudo python main.py
```

if using Anaconda:
```shell
alias sudo='sudo env PATH=$PATH'
sudo python main.py
```

## Multiple sensors support: 
In the beginning of main.py you can add the supported sensors by adding the necessary sensor IDs to variable 'sensors' (line 11).<br/>
Then you can select which sensor to work with by changing the value of 'current_sensor' variable (line 12).
