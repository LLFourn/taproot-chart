#!/usr/bin/env python3

from waitress import serve
import app
import os
import time

def check_loop(check_pid):
    while True:
        time.sleep(60)
        if os.getppid() != check_pid:
            sys.exit(1)
        try:
            app.steal_data()
        except Exception as e:
            print(e)
            time.sleep(600)

parent_pid = os.getpid()
childpid = os.fork()

app.steal_data()

if childpid == 0:
    print("block checking process running as ", parent_pid)
    check_loop(parent_pid)
else:
    serve(app.app.server, host="0.0.0.0", port=8050)
