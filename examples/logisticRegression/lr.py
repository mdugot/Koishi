import koishi
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

dataTypes = ['int','S30','S30','S30','S30','S30','float','float','float','float','float','float','float','float','float','float','float','float','float']
featureNames = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies', 'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions', 'Care_of_Magical_Creatures', 'Charms', 'Flying']

class lr:

    def __init__(self, filename):

        self.data = self.readFeatures(filename)
        koishi.initializeAll()

    def normalize(self, data):
        return (value-minv) / rangev

    def getFeatures(self, rawData):
        data = rawData[:][featureNames].view(float)
        return np.array(data.reshape(len(rawData), len(featureNames)))

    def readFeatures(self, filename):
        rawData = np.genfromtxt(filename, names=True, delimiter=",", dtype=dataTypes)
        return self.getFeatures(rawData)

    def save(self, filename):
        koishi.save(filename, "variable")

    def load(self, filename):
        koishi.load(filename)
