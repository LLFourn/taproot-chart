#!/usr/bin/env python3
import plotly.express as px
import requests as rq
import pandas as pd
import numpy as np
import time

df =  pd.read_csv("data.csv")

while True:
    height = df.iat[-1,0] + 1;
    print(height)
    r = rq.get("https://mempool.space/api/block-height/" + str(height))
    if r.status_code == 200:
        block = r.text
        r = rq.get("https://mempool.space/api/block/" + block)
        signal = (r.json()["version"] & 0x04) >> 2
        new_row = pd.DataFrame(data= { 'height': [height], 'signal' : [signal] })
        df = df.append(new_row)
        df.to_csv("data.csv", index=False)
    elif r.status_code == 404:
        time.sleep(60)
    else:
        print(r.status_code)
        print(r.text)
        time.sleep(600)
