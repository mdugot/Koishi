import matplotlib.pyplot as plt
from data import Dataset

#course = "Care_of_Magical_Creatures"
course = "Arithmancy"
dataset = Dataset("./dataset_train.csv")
dataset.histogram(course, plt)
plt.title(course)
plt.show()

