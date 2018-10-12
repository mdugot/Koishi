#!/usr/bin/python3.6

import lr
import sys

optim = None
if len(sys.argv) > 1:
    optim = sys.argv[1]
model = lr.lr("data.csv")
model.train(1000, 0.1, optim)
model.plotResult()
model.save("save.txt")
