import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv("data.csv")

df['MA'] = df['signal'].rolling(window=100).mean()
d = np.polyfit(df['height'][100:], df['MA'][100:], 1)
f = np.poly1d(d)
df.insert(3, 'line', f(df['height']))
first_height = df[-2016:]['height'][0]
fig = px.line(df[-2016:], x="height", y=["MA","line"], range_y = [0,1], range_x = [ first_height, first_height + 2016])

app.layout = html.Div(children=[
    html.H1(children='Taproot Trend'),

    html.Div(children='''
        A 100 block moving average of signaling
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
