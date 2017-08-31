import os
import web
import visuals
from utils import *
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pyflux as pf 
import threading
from threading import Thread

os.environ["PORT"] = "80"

detailed_output = False

sensors = ['55941031', '2225280']
current_sensor = '2225280'

dbpath = 'data/sensor_'

current_sensor_dbpath = dbpath+current_sensor+".csv"    


render = web.template.render('templates')
        
urls = (
    '/', 'Main',
    '/insert/(.*)', 'Receiver',
    '/last', 'Last_val',
    '/predict_next', 'Predict_next',
    '/predict_next_5', 'Predict_next_5',
    '/reset', 'Reset',
    '/test', 'Test'
)


class Main:
    
    def GET(self):
        
        data = []
        time = []
        
        if os.path.isfile(current_sensor_dbpath):  
        
            df = pd.read_csv(current_sensor_dbpath)
            
            data = df['data'].values
            time = df['time'].values
            
        # uncomment to show only the last 2 days
        
        # if len(data) > 15000:
        #     data = data[len(data)-15000:len(data)]
        #     time = time[len(time)-15000:len(time)]
        
        
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
                            
                            if len(db) < 150000:
                                with open(file_path, 'a') as f:
                                    df.to_csv(f, index = None, header = False)
                                    
                            else:
                                db = db[10000:].reset_index(drop=True)
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
                        
                
                if sensorId == current_sensor:        
                    if os.path.isfile(current_sensor_dbpath):
                        db = pd.read_csv(current_sensor_dbpath)  
                        
                        if len(db) % 500 == False:         # every 500th push from the sensor, which roughly translates to every 30 minutes
                            print "Starting the procedure of autonomous testing..."
                            test = Test()
                            Thread(target = test.GET, kwargs = {'test_fraction': None, 'test_size': 50}).start()        # Running the test procedure in a separate thread
                        
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
        
        max_len = len(x)
        
        if len(x) >= max_len:
            # working on a timeframe of max_len minutes
            x = x[len(x)-max_len:len(x)]
            y = y[len(y)-max_len:len(y)]
        else:
            return "Not enough training data. Currently there is data only for %i minute(s). Please wait until there is data on %i minutes." % (len(x), max_len)
        
        final_series = pd.Series(y, x)
        
        # training ARIMA model
        try:
            model = sm.tsa.statespace.SARIMAX(final_series,
                                            order=(1, 1, 1),
                                            seasonal_order=(0, 0, 0, 5),
                                            enforce_stationarity=True,
                                            enforce_invertibility=False)
            results = model.fit(disp=0)
            
        except ValueError:
            model = sm.tsa.statespace.SARIMAX(final_series,
                                            order=(1, 1, 1),
                                            seasonal_order=(0, 0, 0, 5),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                                            
            print "Warning! The data is heavily skewed, analysis on non-stationary data will be less reliable / accurate. Consider removing skewed parts of the data"
            results = model.fit(disp=0)
        
        n = 10 # how many steps forward to predict
        predicted_vals = results.predict(alpha=0.05, start=1, end=len(final_series)+n).values[len(final_series):]
        predicted = predicted_vals[0]
        
        # array of last n minutes + next n predictions
        y_comb = np.append(y[len(y)-n:], predicted_vals)
        
        indices = np.arange(len(y_comb))
        
        y_comb_normalized = normalize(y_comb)
        
        predicted_normalized = y_comb_normalized[n]     # predicted value normalized in 0-1 with respect to the array of last 10 values and next 10 predictions
        
        # --- calculating the difference for each consecutive observation / prediction --- #
        
        y_comb_diff = []
        
        for i in range(len(y_comb)):
            if i > 0:
                y_comb_diff.append(y_comb[i] - y_comb[i-1])
                
        predicted_change_idx = n - 1
        
        y_comb_diff_normalized = normalize(y_comb_diff)
        
        relative_change = y_comb_diff_normalized[predicted_change_idx]
        
        
        df_acc = pd.read_csv("data/current_accuracy_%s.csv" % current_sensor)
        median_accuracy = df_acc['median_accuracy'].values[-1]
        
        likelihood = (median_accuracy + relative_change) / 2
        
        
        if predicted > y[-1]:
            return "<html>Prediction: the value will <font color='green'>increase</font>.  <br/> Estimated likelihood: <b>%0.2f</b>  <br/><br/> Predicted value for the next minute: <b>%f</b>  <br/>Current value at %s: <b>%f</b>  <br/><br/> Last observation (raw) at %s: <b>%f</b></html>" % (likelihood, predicted, x[-1], y[-1], time_full[-1], vals[-1])
        
        else:
            return "<html>Prediction: the value will <font color='red'>decrease</font>.  <br/> Estimated likelihood: <b>%0.2f</b>  <br/><br/> Predicted value for the next minute: <b>%f</b> <br/>Current value at %s: <b>%f</b>  <br/><br/> Last observation (raw) at %s: <b>%f</b></html>" % (likelihood, predicted, x[-1], y[-1], time_full[-1], vals[-1])
        

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
        
        max_len = len(x)
        
        if len(x) >= max_len:
            # working on a timeframe of max_len minutes
            x = x[len(x)-max_len:len(x)]
            y = y[len(y)-max_len:len(y)]
        else:
            return "Not enough training data. Currently there is data only for %i minute(s). Please wait until there is data on %i minutes." % (len(x), max_len)
        
        final_series = pd.Series(y, x)
        
        # training ARIMA model
        try:
            model = sm.tsa.statespace.SARIMAX(final_series,
                                            order=(1, 1, 1),
                                            seasonal_order=(0, 0, 0, 5),
                                            enforce_stationarity=True,
                                            enforce_invertibility=False)
            results = model.fit(disp=0)
            
        except ValueError:
            model = sm.tsa.statespace.SARIMAX(final_series,
                                            order=(1, 1, 1),
                                            seasonal_order=(0, 0, 0, 5),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                                            
            print "Warning! The data is heavily skewed, analysis on non-stationary data will be less reliable / accurate. Consider removing skewed parts of the data"
            results = model.fit(disp=0)
        
        n = 5 # how many steps forward to predict
        predicted_vals = results.predict(alpha=0.05, start=1, end=len(final_series)+n).values[len(final_series):].tolist()
        
        x_ticks = range(1,n+1)
        
        graph = visuals.plot_data(x_ticks, predicted_vals, current_sensor)
        
        return render.visualize(graph)
        
    def POST(self):
        
        data = web.input(back=[])
            
        if data.back:
            raise web.seeother('/')
            

class Test:
    
    def GET(self, test_fraction = 0.1, test_size = None):
        # example usage: http://127.0.0.1/predict_next
        
        vals = []
        time_full = []
            
        if os.path.isfile(current_sensor_dbpath):
            
            db = pd.read_csv(current_sensor_dbpath)  # checking if the database contains data for a given sensor
        
            vals = db['data'].values.tolist()
            time_full = db['time'].values.tolist()
        
        else:
            return "No data received yet"
          
        # time_full = time_full[19034:]
        # vals = vals[19034:]
            
        time_min = mean_minute_transform(vals, time_full)
                
        x = np.asarray([to_local_time(ts, precision = "minutes") for ts in time_min.keys()])
        y = np.asarray(time_min.values())
        
        max_len = len(x)
        
        if len(x) >= max_len:
            # working on a timeframe of max_len minutes
            x = x[len(x)-max_len:len(x)]
            y = y[len(y)-max_len:len(y)]
        else:
            return "Not enough training data. Currently there is data only for %i minute(s). Please wait until there is data on %i minutes." % (len(x), max_len)
        
        def iteratively_validate(x, y, test_fraction = test_fraction, test_size = test_size):
            
            if test_fraction:
                train_len = int( len(x) * (1 - test_fraction) )
                test_len = int( len(x) * test_fraction )
                
            elif test_size:
                train_len = int( len(x) - test_size )
                test_len = int ( test_size )
            
            print "Fitting the model iteratively %i times" % test_len
            
            hits = 0
            mismatches = 0
            increases = 0
            decreases = 0
            actual_increases = 0
            actual_decreases = 0
            correct_increases = 0
            correct_decreases = 0
            accs = []
            
            predicted_bin = []
            
            for i in range(test_len):
                
                if detailed_output:
                    print "Iteration %i, testing prediction for minute No. %i of the last %i minutes" % (i, i + train_len + 1, max_len)
                
                x_train = x[0:train_len + i]
                y_train = y[0:train_len + i]
                
                y_true = y[train_len + i]
                
                final_series = pd.Series(y_train, x_train)
                
                # training ARIMA model
                try:
                    model = sm.tsa.statespace.SARIMAX(final_series,
                                                    order=(1, 1, 1),
                                                    seasonal_order=(0, 0, 0, 5),
                                                    enforce_stationarity=True,
                                                    enforce_invertibility=False)
                    results = model.fit(disp=0)
                    
                except ValueError:
                    model = sm.tsa.statespace.SARIMAX(final_series,
                                                    order=(1, 1, 1),
                                                    seasonal_order=(0, 0, 0, 5),
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                                                    
                    print "Iteration %i" % i
                    print "Warning! The data is heavily skewed, analysis on non-stationary data will be less reliable / accurate. Consider removing skewed parts of the data"
                    
                    results = model.fit(disp=0)
                                                    
                try:
                    n = 1 # how many steps forward to predict
                    y_pred = results.predict(alpha=0.05, start=1, end=len(final_series)+n).values[-1]
                    
                except:
                    print "\nFailed to fit, numeric error; please wait 1 minute and repeat\n\n"
                    # actual increase
                    if y_true > y_train[-1]: 
                        y_pred = y_train[-1]-0.1
                    if y_true < y_train[-1]: 
                        y_pred = y_train[-1]+0.1

                
                # prediction for an increase
                if y_pred > y_train[-1]: 
                    # actual increase
                    if y_true > y_train[-1]: 
                        hits += 1
                        correct_increases += 1
                        predicted_bin.append(1)
                        
                    else:
                        mismatches += 1
                        predicted_bin.append(0)
                        
                    increases += 1
                        
                # prediction for a decrease`
                else:
                    # actual decrease
                    if y_true < y_train[-1]:
                        hits += 1
                        correct_decreases += 1
                        predicted_bin.append(1)
                        
                    else:
                        mismatches += 1
                        predicted_bin.append(0)
                        
                    decreases += 1
                    
                if y_true > y_train[-1]:
                    actual_increases += 1
                else:
                    actual_decreases += 1
                       
                if detailed_output:
                    acc = float(hits) / float(hits+mismatches)
                    accs.append(acc)
                    print "Accuracy: %s, hits: %s, mismatches: %s" % (acc, hits, mismatches)
                    print
                    print "Number of predicted decreases: %i; Number of predicted increases: %i" % (decreases, increases)
                    print
                    print "Number of correct decreases: %i; Number of correct increases: %i" % (correct_decreases, correct_increases)
                    print
                    print "Number of actual decreases: %i; Number of actual increases: %i" % (actual_decreases, actual_increases)
                    print
                    
                window_size = 20
                    
                if i >= window_size:
                    moving_acc = np.count_nonzero(predicted_bin[i-window_size:i]) / float(window_size)
                    
                    if i % window_size == False:
                        print "Iteration %i" % i
                        print "Moving accuracy (last %i minutes):" % window_size, moving_acc
                        
                    accs.append(moving_acc) # accuracies on sliding windows of 20 minutes
            
            accuracy = float(hits) / float(test_len)
            
            print
            print "Mean moving accuracy (on %i-minute-wide sliding windows): %0.2f" % (window_size, np.mean(accs))
            print "Median moving accuracy (on %i-minute-wide sliding windows): %0.2f" % (window_size, np.median(accs))
            
            data = [[np.median(accs), x[-1]]]
            
            df = pd.DataFrame(data, index = None, columns = ('median_accuracy', 'last_minute'))
            
            if os.path.isfile("data/current_accuracy_%s.csv" % current_sensor):
                with open("data/current_accuracy_%s.csv" % current_sensor, 'a') as f:
                    df.to_csv(f, index = None, header = False)
            
            else:
                with open("data/current_accuracy_%s.csv" % current_sensor, 'w') as f:
                    df.to_csv(f, index = None)
            
            return accuracy, hits, mismatches, accs
            
                
        accuracy, hits, mismatches, accs = iteratively_validate(x, y)
        
        print
        print"Accs: ", accs
        print
        print"Accuracy: %s, hits: %s, mismatches: %s" % (accuracy, hits, mismatches)
        print

        return "Accuracy: %s, hits: %s, mismatches: %s" % (accuracy, hits, mismatches)
        
        

if __name__ == "__main__":
    app = web.application(urls, globals(), autoreload = True)
    app.run()