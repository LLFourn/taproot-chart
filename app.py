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


from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    dcc.Markdown('''
    # Lloyd's Taproot Charts

    Data lifted from [taproot.watch](https://taproot.watch). Code at https://github.com/LLFourn/taproot-chart.
    '''),
    html.Div(id='live-update-text'),
    dcc.Graph(
        id='taproot-chart',
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
              Output('scatter-chart', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_display(n):
    df = pd.read_csv("data.csv")
    style = {'padding': '5px', 'fontSize': '16px'}
    text = [
        html.Ul(children=[
            html.Li(html.Span('Current Height: {}'.format(df.iat[-1,0]), style=style)),
            html.Li(html.Span('{} of the last 100 blocks signaling'.format(df[-100:]['signal'].sum()), style=style)),
            html.Li(html.Span('{:.2f}% chance that we have reached 90%'.format(1- binom.cdf(90,100, df[-100:]['signal'].sum()/100)), style=style))
        ])
    ]

    return text, ma_plot(df), scatter_plot(df)

def scatter_plot(df):
    fig = px.strip(df, x="height", y="miner", color="signal", color_discrete_sequence=["red", "#2CA02C"], title="Green Dot Good, Red Dot Bad")
    fig.update_layout(height=1000)
    fig.update_yaxes(categoryorder='total ascending')
    return fig

def ma_plot(df):
    df['100BlockMA'] = df['signal'].rolling(window=100,min_periods=1).mean()
    d = np.polyfit(df['height'], df['100BlockMA'], 1)
    f = np.poly1d(d)
    last_2016 = df[-2016:].copy()
    last_2016['line'] =  f(last_2016['height'])
    first_height = last_2016.iat[0,0]
    fig = px.line(last_2016, y=["100BlockMA","line"], range_y = [0,1], range_x = [ 0, 2016], color_discrete_sequence=["blue", "#2CA02C"], title="Green Line Go Up (100 block moving average with predicative green line powered by deep learning)")
    fig.update_yaxes(dtick=0.05)
    fig.update_xaxes(dtick=100)
    return fig

def steal_data():
    r = rq.get("https://taproot.watch/blocks")
    if r.status_code == 200:
        json = r.json()
        df = pd.DataFrame([[row['height'],row['miner'],row['signals']] for row in json if 'miner' in row], columns =['height', 'miner', 'signal'])
        df.to_csv("data.csv", index=False)
    else:
        r.raise_for_status()

if __name__ == '__main__':
    steal_data()
    app.run_server(debug=True)
