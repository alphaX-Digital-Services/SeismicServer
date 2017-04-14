import os
import web
import dataset
import visuals
from utils import *
import numpy as np
import pyflux as pf

os.environ["PORT"] = "80"

sensorId = '55941031'

path = 'data/sensors.db'
db = dataset.connect('sqlite:///' + path, engine_kwargs={'connect_args': {'check_same_thread':False}})


render = web.template.render('templates')
        
urls = (
    '/', 'Main',
    '/insert/(.*)', 'Receiver',
    '/last', 'Last_val',
    '/predict_next', 'Predict_next',
    '/predict_next_5', 'Predict_next_5'
)


class Main:
    
    def GET(self):
        
        vals = []
        timestamps = []
            
        if db.tables.count(sensorId):  # checking if the database contains data for the specified sensor
        
            for el in db[sensorId].all():
                vals.append(float(el['data']))
                timestamps.append(el['time'])
            
        graph = visuals.plot_data(timestamps, vals, sensorId)
            
        return render.index(graph)
        
    def POST(self):
        
        data = web.input(predict_next=[], predict_next_5=[])
            
        if data.predict_next:
            raise web.seeother('/predict_next')
            
        elif data.predict_next_5:
            raise web.seeother('/predict_next_5')
            
            
            
class Receiver:        

    def GET(self, received):
        # example usage: http://127.0.0.1/insert/sensorId=2225280&decimal=2.0&timestamp=1490710679

        if received:
            
            sId, decimal, timestamp = getvals(received)
            
            if sensorId == sId:
                
                table = db[sensorId]
                
                table.insert(dict(data = decimal, time = to_local_time(timestamp)))
                
                output_dict = {"type": sensorId, "data": decimal, "time": timestamp}
        
            return "OK / Received: %s" % output_dict
        
        else:
            return "Incorrect format. Please check the format of your GET-request"
        
            
class Last_val:
    
    def GET(self):
        # example usage: http://127.0.0.1/last
        
        if db.tables.count(sensorId):  # checking if the database contains data for a given sensor
            
            decimal = str(db[sensorId].find_one(id=db[sensorId].count())['data'])
            timestamp = to_timestamp(db[sensorId].find_one(id=db[sensorId].count())['time'])
            
            output_dict = {"type": sensorId, "data": decimal, "time": timestamp}
        
            return output_dict
        
        else:
            return "No data received yet"
        
        
class Predict_next:
    
    def GET(self):
        # example usage: http://127.0.0.1/predict_next
        
        vals = []
        time_full = []
            
        if db.tables.count(sensorId):  
        
            for el in db[sensorId].all():
                vals.append(float(el['data']))
                time_full.append(el['time'])
        
        time_min = mean_minute_transform(vals, time_full)
                
        x = np.asarray([to_local_time(ts, precision = "minutes") for ts in time_min.keys()])
        y = np.asarray(time_min.values())
        
        # working on a timeframe of 1000 minutes
        x = x[len(x)-1000:len(x)]
        y = y[len(y)-1000:len(y)]
        
        # using ARIMA model
        model_arima = pf.ARIMA(data=np.asarray(y), ar=3, ma=1, target=np.asarray(x))
        
        # training via maximum likelihood estimation
        trained_arima = model_arima.fit("MLE")
        
        predicted = model_arima.predict(h=1)['Series'].tolist()[0]
        
        if predicted > y[-1]:
            return "Prediction: the value will increase. <br/><br/>Predicted value for the next minute: %f <br/>Current value at %s: %f" % (predicted, x[-1], y[-1])
        
        else:
            return "Prediction: the value will decrease. <br/><br/>Predicted value for the next minute: %f <br/>Current value at %s: %f" % (predicted, x[-1], y[-1])
        

class Predict_next_5:
    
    def GET(self):
        # example usage: http://127.0.0.1/predict_next_5
        
        vals = []
        time_full = []
            
        if db.tables.count(sensorId):  
        
            for el in db[sensorId].all():
                vals.append(float(el['data']))
                time_full.append(el['time'])
        
        time_min = mean_minute_transform(vals, time_full)
                
        x = np.asarray([to_local_time(ts, precision = "minutes") for ts in time_min.keys()])
        y = np.asarray(time_min.values())
        
        # working on a timeframe of 1000 minutes
        x = x[len(x)-1000:len(x)]
        y = y[len(y)-1000:len(y)]
        
        # using ARIMA model
        model_arima = pf.ARIMA(data=np.asarray(y), ar=3, ma=1, target=np.asarray(x))
        
        # training via maximum likelihood estimation
        trained_arima = model_arima.fit("MLE")
        
        predicted = model_arima.predict(h=5, intervals = True)['Series'].tolist()
        
        x_ticks = range(1,6)
        
        graph = visuals.plot_data(x_ticks, predicted, sensorId)
        
        return render.visualize(graph)
        
    def POST(self):
        
        data = web.input(back=[])
            
        if data.back:
            raise web.seeother('/')
        
        

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()