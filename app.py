import dash
import math
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
BLOCKS_IN_A_DAY = 24 * 6
config = {'modeBarButtonsToRemove': ['lasso2d', 'hoverCompareCartesian', 'toggleSpikelines', 'zoomIn2d', 'zoomOut2d', 'select2d'], 'modeBarButtonsToAdd' : ['drawopenpath', 'eraseshape']}

app.layout = html.Div(children=[
    dcc.Markdown('''
    # LLFourn's Taproot Charts

    Data lifted from [mempool.space](https://mempool.space). Available as a csv [here](/assets/data.csv). Spaghetti code at https://github.com/LLFourn/taproot-chart.
    See also [taproot.watch](https://taproot.watch).
    '''),
    html.Div(id='live-update-text'),
    dcc.Graph(
        id='taproot-chart',
        config=config,
    ),
    dcc.Graph(
        id='scatter-chart',
        config=config
    ),
    dcc.Graph(
        id='mining-power-chart',
        config=config
    ),
    dcc.Graph(
        id='inconsistent-chart',
        config=config
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
    df = pd.read_csv("assets/data.csv",index_col='height')
    text = [
        html.Ul(children=[
            html.Li(html.Span('Current Height: {}'.format(df.index[-1]))),
            html.Li(html.Span('{}/2016 completed for period'.format((df.index[-1] - df.index[0] + 1) % 2017 ))),
            html.Li(html.Span('{} of the last 100 blocks signaling'.format(df[-100:]['signal'].sum()))),
            html.Li(html.Span('{:.2f}% chance that we have reached 90%'.format(1- binom.cdf(90,100, df[-100:]['signal'].sum()/100))))
        ])
    ]

    return text, ma_plot(df),miner_dots(df),  mining_power(df), inconsistent_ma(df)

def position_fig(fig,df):
    last_height = df.index[-1]
    first_show_height = max(last_height - 2016, df.index[0])
    last_show_height =  max(last_height, df.index[0] + 2016)
    end_of_period = last_height + (-last_height % 2016)
    counter = end_of_period
    while counter > df.index[0]:
        fig.add_vline(counter)
        counter -= 2016
    fig.update_xaxes(dtick=BLOCKS_IN_A_DAY, tickformat="d", range=[first_show_height, last_show_height])
    fig.update_layout(dragmode='pan')

def ma_plot(df):
    df['100BlockMA'] = df['signal'].rolling(window=100,min_periods=1).mean()
    d = np.polyfit(df.index.values, df['100BlockMA'], 1)
    f = np.poly1d(d)
    # always chart up to the next signaling period
    last_height = df.index[-1]
    end_of_period = last_height - (last_height % 2016) + 2016
    df = df.reindex(np.arange(df.index[0], end_of_period))
    df['line'] = f(df.index)
    fig = px.line(df, y=["100BlockMA","line"], range_y = [0,1],  color_discrete_sequence=["blue", "#2CA02C"])
    fig.update_layout(title={ 'text' : "Number Go Up -- 100 block moving average with predicative green line (powered by deep learning)", 'x': 0.5 })
    fig.update_yaxes(dtick=0.05, title='signal fraction of last 100 blocks')
    fig.add_hline(y=0.9)
    fig.update_layout(height=1000, showlegend=False)
    position_fig(fig,df)

    return fig

def miner_dots(df):
    fig = px.strip(df,x=df.index, y="miner", color="signal", color_discrete_sequence=["red", "#2CA02C"])
    fig.update_layout(height=1000)
    fig.update_yaxes(categoryorder='total ascending', showgrid=True, tickson="boundaries",title=None)
    fig.update_layout(title={ 'text' :"Green Dot Good, Red Dot Bad (dots are blocks)", 'x': 0.5 })
    position_fig(fig,df)
    return fig

def mining_power(df):
    miners = df['miner'].unique()
    block_counter = { name : 0 for name in miners }
    rows = { name : [] for name in miners }

    for (index, row) in df.iterrows():
        block_counter[row.miner] += 1
        window_start = index - 3*BLOCKS_IN_A_DAY
        if df.index[0] <= window_start:
            block_counter[df.loc[window_start]['miner']] -= 1

        for (name, val) in block_counter.items():
            rows[name].append(val)

    color_map = {}
    frac_map = {}
    for name,gdf in df.groupby("miner"):
        # Look over the last 3*BLOCKS_IN_A_DAY blocks
        frac = gdf.loc[df.index[-3*BLOCKS_IN_A_DAY]:]['signal'].mean()
        if np.isnan(frac):
            # Otherwise look at mean for last 10
            frac = gdf['signal'][-10:].mean()
        red = 255
        green = 255
        if frac < 0.5:
            green = frac * 2 * 255
        else:
            red -= frac * 255
        color_map[name] = 'rgba({},{},0, 0.4)'.format(red, green)
        frac_map[name] = frac

    data = pd.DataFrame(data=rows,index=df.index)
    ordered_miners = data.columns.to_list()
    ordered_miners.sort(key=lambda x: (-frac_map[x], -data[x].iloc[-1]))

    fig = go.Figure()
    for name in ordered_miners:
        text = [None for _ in data.index]
        frac_mining_power = data[name].iloc[-1] / (3*BLOCKS_IN_A_DAY)
        if frac_mining_power > 0.01:
            text[math.floor(len(data.index) * 0.9)] = name
        fig.add_trace(go.Scatter(
            name=name,
            x=data.index,
            y=data[name],
            line=dict( color='black', width=0.3  ),
            fillcolor=color_map[name],
            stackgroup='one',
            groupnorm='fraction',
            hoverinfo="name+y",
            mode="lines+text",
            text=text,
            textposition="middle right"
        ))

    fig.update_layout(height=1000)
    fig.update_layout(title={ 'text' : "Miner share of last ~3 days of blocks with color indicating signaling fraction", 'x': 0.5 })
    fig.update_layout(showlegend=False)
    fig.update_yaxes(dtick=0.05, range=[0,1], side='right')
    position_fig(fig,df)

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
    fig.update_layout(title={ 'text' : "Inconsistent Miners who have flucuated in their signaling (25 block moving average of signal fraction)", 'x': 0.5 })
    fig.update_yaxes(dtick=0.1)
    fig.update_yaxes(categoryorder='max ascending', title=None)
    position_fig(fig,df)
    return fig

def check_next_block_mempoolio(df, miner_match):
    if df is None or len(df) < 1:
        height = 681418
        df = pd.DataFrame(columns=['height', 'miner', 'signal'])
    else:
        height = df.iat[-1,0] + 1;
    r = rq.get("https://mempool.space/api/block-height/" + str(height))
    if r.status_code == 200:
        block_id = r.text
        block_info = rq.get("https://mempool.space/api/block/" + block_id).json()
        signal = (block_info["version"] & 0x04) == 0x04
        coinbase_txid = rq.get("https://mempool.space/api/block/" + block_id + "/txid/0").text
        res = rq.get("https://mempool.space/api/tx/" + coinbase_txid)
        res.raise_for_status()
        coinbase_tx = res.json()
        coinbase_address = next(vout['scriptpubkey_address'] for vout in  coinbase_tx['vout'] if vout['scriptpubkey_type'] != 'op_return')
        coinbase_tag = bytes.fromhex(coinbase_tx['vin'][0]['scriptsig']).decode('utf-8', 'replace')
        miner = next((info['name'] for (tag,info) in miner_match['coinbase_tags'].items() if tag in coinbase_tag), None)
        if miner is None:
            miner = miner_match['payout_addresses'][coinbase_address]['name'] or "unknown"
        new_row = pd.DataFrame(data= { 'height': [height], 'signal' : [signal], 'miner' : [miner] })
        print(height, miner, signal)
        df = df.append(new_row)
        df.to_csv("assets/data.csv", index=False)
    elif r.status_code == 404:
        time.sleep(60)
    else:
        print(r.status_code, r.text)
        time.sleep(600)

    return df

# can't get all data from here now https://github.com/hsjoberg/fork-explorer/issues/58
def steal_data_taproot_watch():
    r = rq.get("https://taproot.watch/blocks")
    if r.status_code == 200:
        json = r.json()
        df = pd.DataFrame([[row['height'],row.get('miner') or 'unknown',row['signals']] for row in json if 'signals' in row], columns =['height', 'miner', 'signal'])
        df.to_csv("assets/data.csv", index=False)
    else:
        r.raise_for_status()

if __name__ == '__main__':
#    To actually run in production use waitress_server.py
    app.run_server(debug=True)
