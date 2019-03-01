import matplotlib.pyplot as plt
from data import Dataset

dataset = Dataset("./dataset_train.csv")
dataset.pairPlot()
plt.show()

