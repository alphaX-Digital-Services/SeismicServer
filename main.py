import os
import web
import dataset
import visuals
from utils import *

os.environ["PORT"] = "80"

sensorId = '2225280'

path = 'data/sensors.db'
db = dataset.connect('sqlite:///' + path, engine_kwargs={'connect_args': {'check_same_thread':False}})


render = web.template.render('templates')
        
urls = (
    '/', 'Main',
    '/insert/(.*)', 'Receiver',
    '/last', 'Last_val'
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
        
        

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()