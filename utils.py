import pandas as pd
from collections import OrderedDict
import time


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
    
    

def to_local_time(timestamp, precision = "seconds"):

    local_time = time.localtime(int(timestamp))
    
    if precision == "seconds":
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        
    elif precision == "minutes":
        local_time = time.strftime("%Y-%m-%d %H:%M", local_time)
    
    return str(local_time)
    
    

def to_timestamp(local_time, output_format = 'str'):
    
    time_tuple = time.strptime(local_time, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(time_tuple)
    
    if output_format == 'str':
        return str(int(timestamp))
        
    elif output_format == 'int':
        return int(timestamp)
        
        
        
def round_timestamp(timestamp, precision = "minutes"):
    
    """
    Takes timestamp as an input, converts it to datetime, rounds to the nearest minute / hour, converts it back to timestamp format
    
    Examples: 
    
    Timestamp 1491280347 stands for 2017-04-04 10:32:27, and it rounds to 2017-04-04 10:32:00, which is 1491280320
    Timestamp 1491280351 stands for 2017-04-04 10:32:31, and it rounds to 2017-04-04 10:33:00, which is 1491280380
    
    Usage:
    
    >>> round_timestamp('1491280347', precision = "minutes")
    >>> '1491280320'
    
    >>> round_timestamp('1491280351', precision = "minutes")
    >>> '1491280380'
    
    """
    
    full_datetime = to_local_time(timestamp)
    
    if precision == "minutes":
        rounded = to_timestamp(str(pd.Timestamp(full_datetime).round('min')))
    
    elif precision == "hours":
        rounded = to_timestamp(str(pd.Timestamp(full_datetime).round('h')))
        
    else:
        raise AssertionError("Precision should be either 'minutes' or 'hours'", distance)

    return rounded
    
    
    
def mean_minute_transform(vals, time_full):
    
    """ 
    
    Transforms (local) datetime values of format '2017-04-04 10:32:02' to timestamps,
    converts it to the nearest minutes in timestamp representation,
    and converts sensor data to values averaged over minute-wide intervals.
    
    Input: 
    vals, array of float values (sensor data)
    time_full, array of datetime values of format '2017-04-04 10:32:02'
    
    Output:
    OrderedDict object with timestamps rounded to minutes as keys and minute-wide averages as values
    
    """
    
    time_min = OrderedDict()
    
    for idx, t in enumerate(time_full):
        
        ts = to_timestamp(t, output_format = 'int')
        
        # ticks are in minutes
        tick = int(round_timestamp(ts, precision = "minutes"))
        last_received_tick = to_timestamp(time_full[-1], output_format = 'int')    # will be the timestamp as from /last: to_timestamp(db[sensorId].find_one(id=db[sensorId].count())['time'])
        
        if time_min:
            if tick in time_min:
                # summing the previous measurement with a new measurement within this particular minute
                prev_val = time_min[tick]
                time_min.update({tick: (prev_val + vals[idx])})
                
                counter += 1
                
                prev_tick = tick
                
                # ToDo: rewrite
                if ts == last_received_tick:
                    sum_last_tick = time_min[tick]
                    time_min.update({tick: (sum_last_tick / counter)})
            
            else:
                # dividing the sum for the previous minute by the number of measurements, to get the average for that minute
                sum_prev_tick = time_min[prev_tick]
                time_min.update({prev_tick: (sum_prev_tick / counter)})
                    
                # registering new tick (minute)
                tick_no += 1
                
                # resetting counter of measurements within 1 minute, as a new minute / tick just started
                counter = 1
                
                time_min.update({tick: vals[idx]})
            
        # starting tick
        else:
            time_min.update({tick: vals[idx]})
            tick_no = 0
            
            # counts number of datapoints / measurements within 1 minute
            counter = 1
            
            prev_tick = tick
            
    return time_min
