#!/usr/bin/env python3

from waitress import serve
import app
import os
import time
import pandas as pd
import sys
import requests as rq

def check_loop_taproot_watch(check_pid):
    app.steal_data()
    while True:
        time.sleep(60)
        if os.getppid() != check_pid:
            sys.exit(1)
        try:
            app.steal_data_taproot_watch()
        except Exception as e:
            print(e)
            time.sleep(600)

def check_loop_mempoolio(check_pid):
    miner_match = rq.get("https://raw.githubusercontent.com/0xB10C/known-mining-pools/master/pools.json").json()
    try:
        df =  pd.read_csv("data.csv")
    except:
        df = None
    while True:
        if os.getppid() != check_pid:
            sys.exit(0)
        try:
            df = app.check_next_block_mempoolio(df, miner_match)
        except Exception as e:
            print("Failed to update:", repr(e))
            time.sleep(60)

parent_pid = os.getpid()
childpid = os.fork()


if childpid == 0:
    print("block checking process running as ", parent_pid)
    check_loop_mempoolio(parent_pid)
else:
    serve(app.app.server, host="0.0.0.0", port=8050)
