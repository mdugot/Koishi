import matplotlib.pyplot as plt
from data import Dataset

c1 = "Astronomy"
c2 = "Defense_Against_the_Dark_Arts"
dataset = Dataset("./dataset_train.csv")
dataset.scatterPlot(c1, c2, plt)
plt.xlabel(c1)
plt.ylabel(c2)
plt.show()

