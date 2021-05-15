#!/usr/bin/env python3
import app
import requests as rq
import sys
import time
import pandas as pd
import os


def check_loop_taproot_watch(check_pid):
    steal_data_taproot_watch()
    while True:
        time.sleep(60)
        if os.getppid() != check_pid:
            sys.exit(1)
        try:
            steal_data_taproot_watch()
        except Exception as e:
            print(e)
            time.sleep(600)

def check_loop_mempoolio(check_pid):
    miner_match = rq.get("https://raw.githubusercontent.com/0xB10C/known-mining-pools/master/pools.json").json()
    try:
        df =  pd.read_csv("assets/data.csv")
    except:
        df = None
    while True:
        if os.getppid() != check_pid:
            sys.exit(0)
        try:
            df = check_next_block_mempoolio(df, miner_match)
        except Exception as e:
            print("Failed to update:", repr(e))
            time.sleep(60)


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
    check_loop_mempoolio(os.getppid())
