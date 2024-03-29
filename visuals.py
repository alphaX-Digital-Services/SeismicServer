from plotly.offline import plot
import plotly.graph_objs as go


def plot_data(timestamps, data, sensorId):
    
    scattered = [go.Scatter(x=timestamps, y=data)]
    
    layout = dict(
        title = "Sensor ID: %s" % sensorId, height = 800, width = 1480
    )
    
    fig = dict(data=scattered, layout=layout)

    output = plot(fig, output_type = 'div')
    
    return output