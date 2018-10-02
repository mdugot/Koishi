#!/usr/bin/python3.6

import lr
import sys
if len(sys.argv) <= 1:
    print("Enter a value in km to predict.")
    quit()
model = lr.lr("data.csv")
model.load("save.txt")
km = float(sys.argv[1])
model.plotResult(km)
e = model.predict(km)
print("Price estimation for " + str(km) + " km : " + str(e))

