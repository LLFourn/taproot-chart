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
    # LLFourn's Taproot Charts

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
    df = pd.read_csv("data.csv",index_col='height')
    text = [
        html.Ul(children=[
            html.Li(html.Span('Current Height: {}'.format(df.index[-1]))),
            html.Li(html.Span('{}/2016 completed for period'.format((df.index[-1] - df.index[0] + 1) % 2017 ))),
            html.Li(html.Span('{} of the last 100 blocks signaling'.format(df[-100:]['signal'].sum()))),
            html.Li(html.Span('{:.2f}% chance that we have reached 90%'.format(1- binom.cdf(90,100, df[-100:]['signal'].sum()/100))))
        ])
    ]

    return text, ma_plot(df), scatter_plot(df)

def scatter_plot(df):
    fig = px.strip(df, y="miner", x=df.index, color="signal", color_discrete_sequence=["red", "#2CA02C"])
    fig.update_layout(height=1000)
    fig.update_xaxes(dtick=24*6, tickformat="d")
    fig.update_yaxes(categoryorder='total ascending', showgrid=True, tickson="boundaries")
    fig.update_layout(title={ 'text' :"Green Dot Good, Red Dot Bad", 'x': 0.5 })
    return fig

def ma_plot(df):
    df['100BlockMA'] = df['signal'].rolling(window=100,min_periods=1).mean()
    d = np.polyfit(df.index.values, df['100BlockMA'], 1)
    f = np.poly1d(d)
    last_2016 = df[-2016:].copy()
    first_height = last_2016.index[0]
    last_2016 = last_2016.reindex(np.arange(first_height, first_height + 2016))
    last_2016['line'] = f(last_2016.index)
    fig = px.line(last_2016, y=["100BlockMA","line"], range_y = [0,1],  color_discrete_sequence=["blue", "#2CA02C"])
    fig.update_layout(title={ 'text' : "Number Go Up -- 100 block moving average with predicative green line (powered by deep learning)", 'x': 0.5 })
    fig.update_yaxes(dtick=0.05)
    fig.update_xaxes(dtick=24*6, tickformat="d")
    fig.add_hline(y=0.9)
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
    # To actually run in production use waitress_server.py
    steal_data()
    app.run_server(debug=True)
