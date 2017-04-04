import time

def to_local_time(timestamp):

    local_time = time.localtime(int(timestamp))
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    
    return local_time

def to_timestamp(local_time):
    
    time_tuple = time.strptime(local_time, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(time_tuple)
    
    return str(int(timestamp))
    

def getvals(data):
    # data: sensorId=2225280&decimal=1.22377&timestamp=1490710679

    s_end = data.find('&')
    
    sensorId = str(data[9:s_end])
    
    
    d_start = data.find('decimal=')+8
    d_end = data[d_start:].find('&')
    
    decimal = str(data[d_start : d_start+d_end])
    
    
    t_start = data.find('timestamp=')+10
    
    timestamp = str(int(int(data[t_start:])/1000))            # converting milliseconds' timestamps to seconds' timestamps

    return sensorId, decimal, timestamp