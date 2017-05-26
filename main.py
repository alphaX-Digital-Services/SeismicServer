import os
import web
import visuals
from utils import *
import numpy as np
import pandas as pd
import pyflux as pf

os.environ["PORT"] = "80"

sensors = ['55941031']
current_sensor = '55941031'

dbpath = 'data/sensor_'

current_sensor_dbpath = dbpath+current_sensor+".csv"    


render = web.template.render('templates')
        
urls = (
    '/', 'Main',
    '/insert/(.*)', 'Receiver',
    '/last', 'Last_val',
    '/predict_next', 'Predict_next',
    '/predict_next_5', 'Predict_next_5',
    '/reset', 'Reset'
)


class Main:
    
    def GET(self):
        
        data = []
        time = []
        
        if os.path.isfile(current_sensor_dbpath):  
        
            df = pd.read_csv(current_sensor_dbpath)
            
            data = df['data'].values
            time = df['time'].values
            
        
        if len(data) > 35000:
            data = data[len(data)-35000:len(data)]
            time = time[len(time)-35000:len(time)]
        
        
        if os.path.isfile("data/reset_timevals.csv"):  
            reset_timevals = pd.read_csv("data/reset_timevals.csv")

            reset_idx = np.where(time == reset_timevals['time'].values[-1])[0][0]
            
            time = time[reset_idx:]
            data = data[reset_idx:]
        
        graph = visuals.plot_data(time, data, current_sensor)
            
        return render.index(graph)
        
    def POST(self):
        
        data = web.input(predict_next=[], predict_next_5=[], reset=[], restore=[])
            
        if data.predict_next:
            raise web.seeother('/predict_next')
            
        elif data.predict_next_5:
            raise web.seeother('/predict_next_5')
            
        elif data.reset:
                
            if os.path.isfile(current_sensor_dbpath): 
            
                df = pd.read_csv(current_sensor_dbpath)
                
                time = df['time'].values
                
            data = [[current_sensor, time[-1]]]
            
            df = pd.DataFrame(data, index = None, columns = ('sensorId', 'time'))
            
            if os.path.isfile("data/reset_timevals.csv"):
                with open("data/reset_timevals.csv", 'a') as f:
                    df.to_csv(f, index = None, header = False)
            
            else:
                with open("data/reset_timevals.csv", 'w') as f:
                    df.to_csv(f, index = None)
                
            raise web.seeother('/')
            
        elif data.restore:
            os.remove("data/reset_timevals.csv")
            raise web.seeother('/')
            
            
            
class Receiver:        

    def GET(self, received):
        # example usage: http://127.0.0.1/insert/sensorId=55941031&decimal=2.0&timestamp=1490710679

        if received:
            
            sensorId, decimal, timestamp = getvals(received)
            
            if sensorId in sensors:
                
                data = [[to_local_time(timestamp), decimal]]
                
                file_path = dbpath + sensorId + ".csv"
                temp_path = dbpath + sensorId + ".tmp"
                
                try:
                    
                    # adding temp/buffer data that accumulated while the file was open by another process
                    if os.path.isfile(temp_path):
                        
                        buffer = pd.read_csv(temp_path).values.tolist()
                        buffer.append(data)
                        
                        df = pd.DataFrame(buffer, index = None, columns = ('time', 'data'))
                        
                        with open(file_path, 'a') as f:
                            df.to_csv(f, index = None, header = False)
                        
                    else:
                        
                        df = pd.DataFrame(data, index = None, columns = ('time', 'data'))
                        
                        if os.path.isfile(file_path):
                            
                            db = pd.read_csv(file_path)
                            
                            if len(db) < 70000:
                                with open(file_path, 'a') as f:
                                    df.to_csv(f, index = None, header = False)
                                    
                            else:
                                db = db[35000:].reset_index(drop=True)
                                with open(file_path, 'w') as f:
                                    db.to_csv(f, index = None)
                                    
                                with open(file_path, 'a') as f:
                                    df.to_csv(f, index = None, header = False)
                        
                        else:
                            with open(file_path, 'w') as f:
                                df.to_csv(f, index = None)
                
                # if the file is open by another process
                except IOError:
                    
                    if os.path.isfile(temp_path):
                        with open(temp_path, 'a') as f:
                            df.to_csv(f, index = None, header = False)
                    
                    else:
                        with open(temp_path, 'w') as f:
                            df.to_csv(f, index = None)
                        
                output_dict = {"type": sensorId, "data": decimal, "time": timestamp}
                
                return "OK / Received: %s" % output_dict
                
            else:
                return "<html>Wrong sensor ID value. </br>Sensor ID should be one of the following: %s </br>Value entered: %s</html>" % (" ".join(x for x in sensors), sensorId)
        
        else:
            return "Incorrect format. Please check the format of your GET-request"
        
            
class Last_val:
    
    def GET(self):
        # example usage: http://127.0.0.1/last
        
        if os.path.isfile(current_sensor_dbpath):
            
            db = pd.read_csv(current_sensor_dbpath)  # checking if the database contains data for a given sensor
        
            value = db['data'].values[-1]
            time = db['time'].values[-1]
            
            output_dict = {"type": current_sensor, "data": str(value), "timemestamp": to_timestamp(time), "local_time": time}
        
            return output_dict
        
        else:
            return "No data received yet"
        
        
class Predict_next:
    
    def GET(self):
        # example usage: http://127.0.0.1/predict_next
        
        vals = []
        time_full = []
            
        if os.path.isfile(current_sensor_dbpath):
            
            db = pd.read_csv(current_sensor_dbpath)  # checking if the database contains data for a given sensor
        
            vals = db['data'].values.tolist()
            time_full = db['time'].values.tolist()
        
        else:
            return "No data received yet"
            
        time_min = mean_minute_transform(vals, time_full)
                
        x = np.asarray([to_local_time(ts, precision = "minutes") for ts in time_min.keys()])
        y = np.asarray(time_min.values())
        
        if len(x) > 1000:
            # working on a timeframe of 1000 minutes
            x = x[len(x)-1000:len(x)]
            y = y[len(y)-1000:len(y)]
        else:
            return "Not enough training data. Currently there is data only for %i minute(s). Please wait until there is data on 1000 minutes." % len(x)
        
        # using ARIMA model
        model_arima = pf.ARIMA(data=np.asarray(y), ar=3, ma=1, target=np.asarray(x))
        
        # training via maximum likelihood estimation
        trained_arima = model_arima.fit("MLE")
        
        predicted = model_arima.predict(h=1)['Series'].tolist()[0]
        
        if predicted > y[-1]:
            return "<html>Prediction: the value will increase. <br/><br/>Predicted value for the next minute: %f <br/>Current value at %s: %f <br/><br/>Last observation (raw) at %s: %f</html>" % (predicted, x[-1], y[-1], time_full[-1], vals[-1])
        
        else:
            return "<html>Prediction: the value will decrease. <br/><br/>Predicted value for the next minute: %f <br/>Current value at %s: %f <br/><br/>Last observation (raw) at %s: %f</html>" % (predicted, x[-1], y[-1], time_full[-1], vals[-1])
        

class Predict_next_5:
    
    def GET(self):
        # example usage: http://127.0.0.1/predict_next_5
        
        if os.path.isfile(current_sensor_dbpath):
            
            db = pd.read_csv(current_sensor_dbpath)  # checking if the database contains data for a given sensor
        
            vals = db['data'].values.tolist()
            time_full = db['time'].values.tolist()
        
        else:
            return "No data received yet"
        
        time_min = mean_minute_transform(vals, time_full)
                
        x = np.asarray([to_local_time(ts, precision = "minutes") for ts in time_min.keys()])
        y = np.asarray(time_min.values())
        
        if len(x) > 1000:
            # working on a timeframe of 1000 minutes
            x = x[len(x)-1000:len(x)]
            y = y[len(y)-1000:len(y)]
        else:
            return "Not enough training data. Currently there is data only for %i minute(s). Please wait until there is data on 1000 minutes." % len(x)
        
        # using ARIMA model
        model_arima = pf.ARIMA(data=np.asarray(y), ar=3, ma=1, target=np.asarray(x))
        
        # training via maximum likelihood estimation
        trained_arima = model_arima.fit("MLE")
        
        predicted = model_arima.predict(h=5, intervals = True)['Series'].tolist()
        
        x_ticks = range(1,6)
        
        graph = visuals.plot_data(x_ticks, predicted, current_sensor)
        
        return render.visualize(graph)
        
    def POST(self):
        
        data = web.input(back=[])
            
        if data.back:
            raise web.seeother('/')
        
        

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()