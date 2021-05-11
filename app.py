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
import plotly.graph_objects as go
import random


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
    dcc.Graph(
        id='mining-power-chart'
    ),
    dcc.Graph(
        id='inconsistent-chart'
    ),
    dcc.Interval(
        id='interval-component',
        interval=10*60*1000, # in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('live-update-text', 'children'),
              Output('taproot-chart', 'figure'),
              Output('scatter-chart', 'figure'),
              Output('mining-power-chart', 'figure'),
              Output('inconsistent-chart', 'figure'),
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

    return text, ma_plot(df), scatter_plot(df),  mining_power(df), inconsistent_ma(df),

def scatter_plot(df):
    fig = px.strip(df,x=df.index, y="miner", color="signal", color_discrete_sequence=["red", "#2CA02C"])
    fig.update_layout(height=1000)
    fig.update_xaxes(dtick=24*6, tickformat="d")
    fig.update_yaxes(categoryorder='total ascending', showgrid=True, tickson="boundaries",title=None)
    fig.update_layout(title={ 'text' :"Green Dot Good, Red Dot Bad (dots are blocks)", 'x': 0.5 })
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
    fig.update_yaxes(dtick=0.05, title='signal fraction of last 100 blocks')
    fig.update_xaxes(dtick=24*6, tickformat="d")
    fig.add_hline(y=0.9)
    fig.add_vline(first_height + 2016)
    fig.update_layout(height=1000, showlegend=False)
    return fig

def mining_power(df):
    miners = df['miner'].unique()
    block_counter = { name : 0 for name in miners }
    rows = { name : [] for name in miners }

    for (index, row) in df.iterrows():
        block_counter[row.miner] += 1
        window_start = index - 300
        if df.index[0] <= window_start:
            block_counter[df.loc[window_start]['miner']] -= 1

        for (name, val) in block_counter.items():
            rows[name].append(val)

    color_map = {}
    frac_map = {}
    for name,gdf in df.groupby("miner"):
        # Look over the last 500 blocks
        frac = gdf.loc[df.index[-300]:]['signal'].mean()
        if np.isnan(frac):
            # If they haven't produced any in last 500 look at overall mean
            frac = gdf['signal'][-10:].mean()
        red = 1
        green = 1
        if frac < 0.5:
            green = frac * 2 * 255.0
        else:
            red -= frac
        color_map[name] = 'rgba({:.3f},{:.3f},0, 0.4)'.format(red, green)
        frac_map[name] = frac

    data = pd.DataFrame(data=rows,index=df.index)
    ordered_miners = data.columns.to_list()
    ordered_miners.sort(key=lambda x: (-frac_map[x], -data[x].iloc[-1]))

    fig = go.Figure()
    for name in ordered_miners:
        fig.add_trace(go.Scatter(
            name=name,
            x=data.index,
            y=data[name],
            line=dict( color='black', width=0.3  ),
            fillcolor=color_map[name],
            stackgroup='one',
            groupnorm='fraction',
            hoverinfo="name+y"
        ))


    fig.update_xaxes(dtick=24*6, tickformat="d")
    fig.update_layout(height=1000)
    fig.update_layout(title={ 'text' : "Miner share of last 300 blocks with color based on signaling fraction of their blocks", 'x': 0.5 })
    fig.update_layout(showlegend=False)
    fig.update_yaxes(dtick=0.05, range=[0,1], side='right')

    return fig

def inconsistent_ma(df):
    grouped = df.groupby("miner")
    inconsistent = []
    for name,gdf in grouped:
        gdf[name] = gdf['signal'].rolling(window=20, min_periods=5).mean()
        signaling = gdf[ gdf['signal'] == True ]
        if len(signaling) > 0:
            if not gdf.loc[signaling.index[0]:]['signal'].all():
                inconsistent.append(gdf[name])

    data = pd.concat(inconsistent,axis=1).fillna(method='ffill')
    fig = px.line(data, range_y = [0,1])
    fig.update_layout(title={ 'text' : "Inconsistent Miners who have didn't signal after their first signal (25 block moving average of signal fraction)", 'x': 0.5 })
    fig.update_yaxes(dtick=0.1)
    fig.update_xaxes(dtick=24*6, tickformat="d")
    fig.update_yaxes(categoryorder='max ascending', title=None)
    return fig


def steal_data():
    r = rq.get("https://taproot.watch/blocks")
    if r.status_code == 200:
        json = r.json()
        df = pd.DataFrame([[row['height'],row.get('miner') or 'unknown',row['signals']] for row in json if 'signals' in row], columns =['height', 'miner', 'signal'])
        df.to_csv("data.csv", index=False)
    else:
        r.raise_for_status()

if __name__ == '__main__':
    # To actually run in production use waitress_server.py
    steal_data()
    app.run_server(debug=True)
