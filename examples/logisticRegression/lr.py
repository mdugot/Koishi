import koishi
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

dataTypes = ['int','S30','S30','S30','S30','S30','float','float','float','float','float','float','float','float','float','float','float','float','float']
featureNames = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies', 'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions', 'Care_of_Magical_Creatures', 'Charms', 'Flying']
shortnames = ['Arithmancy', 'Astronomy', 'Herbology', 'DADA', 'Divination', 'MS', 'AR', 'HoM', 'Transf.', 'Potions', 'CMC', 'Charms', 'Flying']

class lr:

    def __init__(self, filename):

        self.data = np.nan_to_num(self.readFeatures(filename))
        self.kdata = koishi.Tensor(self.data).transpose([1,0])
        #self.normalize()
        self.description = {}
        self.description["Count"] = []
        self.description["Mean"] = []
        self.description["Std"] = []
        self.description["Min"] = []
        self.description["25%"] = []
        self.description["50%"] = []
        self.description["75%"] = []
        self.description["Max"] = []
        for idx in range(len(featureNames)):
            self.description["Count"].append(self.kdata[idx].count())
            self.description["Mean"].append(self.kdata[idx].mean())
            self.description["Std"].append(self.kdata[idx].std())
            self.description["Min"].append(self.kdata[idx].min())
            self.description["25%"].append(self.kdata[idx].percentile(25))
            self.description["50%"].append(self.kdata[idx].percentile(50))
            self.description["75%"].append(self.kdata[idx].percentile(75))
            self.description["Max"].append(self.kdata[idx].max())
        koishi.initializeAll()

    def normalize(self):
        shape = [int(self.kdata.shape().eval()[0])]
        print(shape)
        splitData = self.kdata.split(0)
        normalizeData = splitData.substract(splitData.mean()).divide(splitData.range())
        self.kdata = normalizeData.merge(shape)


    def describe(self):
        padding = 12
        precision = 4
        print("\t", end="")
        for name in shortnames:
            print("%-*s "%(padding,name), end="")
        print("")
        for name,description in self.description.items():
            print("%s\t"%name, end="")
            for value in description:
                print("%-*.*f "%(padding,precision,value.eval()), end="")
            print("")

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

if __name__ == "__main__":
    data = lr("dataset_train.csv")
    data.describe()
    data.normalize()
    print(data.kdata.mean())
