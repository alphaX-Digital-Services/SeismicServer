# SeismicServer

SeismicServer Statistical Prediction App receives a stream of data in the following format: <br/>
sensorId=2225280&decimal=2.0&timestamp=1490710679 <br/>

The received data is visualized as a time-series chart and stored for further statistical predictions: one-step-ahead or n-steps-ahead predictions. <br/>

*Main interface with a chart and prediction functionality:* http://127.0.0.1/ <br/>
*Example input request [insert new value]:* http://127.0.0.1/insert/sensorId=2225280&decimal=2.0&timestamp=1490710679 <br/>
*Example output request [return last value]:* http://127.0.0.1/last <br/>
*Example prediction request [to be implemented in the next versions]:* http://127.0.0.1/predict_next <br/>
 <br/>
 <br/>
Screenshot:

![SeismicServer Statistical Prediction App, alpha version](/data/screenshot_alpha.png?raw=true "SeismicServer alpha")
