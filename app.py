import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys
import requests as rq
import time
from scipy.stats import binom

from datetime import datetime

from dash.dependencies import Input, Output
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


app.layout = html.Div(children=[
    html.H1(children='Red Line Good. Number Go Up.'),
    html.Div(id='live-update-text'),
    dcc.Graph(
        id='taproot-chart',
    ),
    dcc.Graph(
        id='binom-chart',
    ),
    dcc.Graph(
        id='scatter-chart'
    ),
    dcc.Interval(
        id='interval-component',
        interval=60*1000, # in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('live-update-text', 'children'),
              Output('taproot-chart', 'figure'),
              Output('binom-chart', 'figure'),
              Output('scatter-chart', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    df = pd.read_csv("data.csv")
    style = {'padding': '5px', 'fontSize': '16px'}
    text = [
        html.Ul(children=[
            html.Li(html.Span('Current Height: {}'.format(df.iat[-1,0]), style=style)),
            html.Li(html.Span('{} of the last 100 blocks signaling'.format(df[-100:]['signal'].sum()), style=style)),
            html.Li(html.Span('{:.2f}% that we have reached 90%'.format(1- binom.cdf(90,100, df[-100:]['signal'].sum()/100)), style=style))
        ])
    ]


    return text, ma_plot(df), binom_plot(df), scatter_plot(df)

def scatter_plot(df):
    fig = px.strip(df, x="height", color="signal")
    return fig

def ma_plot(df):
    df['100BlockMA'] = df['signal'].rolling(window=100,min_periods=1).mean()
    d = np.polyfit(df['height'], df['100BlockMA'], 1)
    f = np.poly1d(d)
    last_2016 = df[-2016:].copy()
    last_2016['line'] =  f(last_2016['height'])
    first_height = last_2016.iat[0,0]
    fig = px.line(last_2016, y=["100BlockMA","line"], range_y = [0,1], range_x = [ 0, 2016])
    fig.update_yaxes(dtick=0.05)
    fig.update_xaxes(dtick=100)
    return fig

def binom_plot(df):
    data = [ [i, 1- binom.cdf(i,100, df[-100:]['signal'].sum()/100), 1- binom.cdf(i*5,500, df[-500:]['signal'].sum()/500),] for i in range(10,91) ]
    dist = pd.DataFrame(data, columns = ['%signal', 'last 100', 'last 500'])
    binom_plot = px.line(dist, x='%signal', y=['last 100', 'last 500'])
    return binom_plot


def check_next_block(df):
    height = df.iat[-1,0] + 1;
    r = rq.get("https://mempool.space/api/block-height/" + str(height))
    if r.status_code == 200:
        block = r.text
        r = rq.get("https://mempool.space/api/block/" + block)
        signal = (r.json()["version"] & 0x04) >> 2
        print(r.text)
        print(datetime.now(), height, block, signal)
        new_row = pd.DataFrame(data= { 'height': [height], 'signal' : [signal] })
        df = df.append(new_row)
        df.to_csv("data.csv", index=False)
    elif r.status_code == 404:
        time.sleep(60)
    else:
        print(r.status_code)
        print(r.text)
        time.sleep(600)

    return df


def check_loop(check_pid):
    df =  pd.read_csv("data.csv")
    while True:
        if os.getppid() != check_pid:
            sys.exit(1)
        try:
            df = check_next_block(df)
        except Exception as e:
            print(e)
            time.sleep(60)


if __name__ == '__main__':
    check_loop(os.getppid())
#    app.run_server(debug=True)
