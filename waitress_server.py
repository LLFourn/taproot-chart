#!/usr/bin/env python3

from waitress import serve
import app
import grab_data
import os

parent_pid = os.getpid()
childpid = os.fork()


if childpid == 0:
    print("block checking process running as ", parent_pid)
    grab_data.check_loop_mempoolio(parent_pid)
else:
    serve(app.app.server, host="0.0.0.0", port=8050)
