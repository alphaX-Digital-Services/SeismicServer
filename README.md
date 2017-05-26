# SeismicServer

SeismicServer Statistical Prediction App receives a stream of data in the following format: <br/>
sensorId=55941031&decimal=2.0&timestamp=1490710679 <br/>

The received data is visualized as a time-series chart and stored for further statistical predictions: one-step-ahead or n-steps-ahead predictions. <br/>

*Main interface with a chart and prediction functionality:* http://127.0.0.1/ <br/>
*Example input request [insert new value]:* http://127.0.0.1/insert/sensorId=55941031&decimal=2.0&timestamp=1490710679 <br/>
*Example output request [return last value]:* http://127.0.0.1/last <br/>
*Example one-step-ahead prediction request:* http://127.0.0.1/predict_next <br/>
*Example 5-steps-ahead prediction request:* http://127.0.0.1/predict_next_5 <br/>
 <br/>
 <br/>

![SeismicServer Statistical Prediction App, beta version](/data/seismicserver_beta.png?raw=true "SeismicServer beta")

## Depdendencies

For Ubuntu 16.04 with Python 2.7.13 with SciPy stack (or Anaconda2):

```shell

pip install web.py

pip install plotly

sudo apt-get install xvfb libav-tools xorg-dev libsdl2-dev swig cmake

pip install pyflux
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
