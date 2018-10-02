#!/usr/bin/python3.6

import lr
model = lr.lr("data.csv")
model.train(1000, 0.1)
model.plotResult()
model.save("save.txt")
