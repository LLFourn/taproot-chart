#!/usr/bin/env python3

from waitress import serve
import app
import os

parent_pid = os.getpid()
childpid = os.fork()

if childpid == 0:
    print("block checking process running as ", parent_pid)
    app.check_loop(parent_pid)
else:
    serve(app.app.server, host="0.0.0.0", port=8050)
