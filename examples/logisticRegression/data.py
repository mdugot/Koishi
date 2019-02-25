import koishi
import random
from tqdm import tqdm
import numpy as np
if np.__version__ >= '1.16':
    import numpy.lib.recfunctions as rf
from tqdm import tqdm
import matplotlib.pyplot as plt

dataTypes = ['int','S30','S30','S30','S30','S30','float','float','float','float','float','float','float','float','float','float','float','float','float']
featureNames = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies', 'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions', 'Care_of_Magical_Creatures', 'Charms', 'Flying']
shortnames = ['Arithmancy', 'Astronomy', 'Herbology', 'DADA', 'Divination', 'MS', 'AR', 'HoM', 'Transf.', 'Potions', 'CMC', 'Charms', 'Flying']
houseColumn = "Hogwarts_House"
houseNames = ['Ravenclaw', 'Hufflepuff', 'Gryffindor', 'Slytherin']
houseColor = {
    "Ravenclaw": 'blue',
    "Hufflepuff": 'yellow',
    "Gryffindor": 'red',
    "Slytherin": 'green'}

class Data:

    def __init__(self, filename):

        data, houses = self.readData(filename)

        self.dataByStudent = koishi.Tensor(data)
        self.dataByCourse = self.dataByStudent.transpose([1,0])
        self.houses = {}
        self.debug = houses
        for name in houseNames:
            self.houses[name] = [int(idx) for idx in np.where(houses == bytes(name, "utf-8"))[0]]

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
            self.description["Count"].append(self.dataByCourse[idx].count())
            self.description["Mean"].append(self.dataByCourse[idx].mean())
            self.description["Std"].append(self.dataByCourse[idx].std())
            self.description["Min"].append(self.dataByCourse[idx].min())
            self.description["25%"].append(self.dataByCourse[idx].percentile(25))
            self.description["50%"].append(self.dataByCourse[idx].percentile(50))
            self.description["75%"].append(self.dataByCourse[idx].percentile(75))
            self.description["Max"].append(self.dataByCourse[idx].max())

        self.dataByCourse = self.normalize(self.dataByCourse)
        self.trainData = self.dataByCourse.transpose([1,0])
        self.dataByStudent = self.dataByStudent.split(0)
        koishi.initializeAll()

        self.houseDataByStudent = {}
        self.houseDataByCourse = {}
        for name in houseNames:
            self.houseDataByStudent[name] = self.dataByStudent.take(self.houses[name]).merge([len(self.houses[name])])
            self.houseDataByCourse[name] = self.houseDataByStudent[name].transpose([1,0])

    def normalize(self, data):
        shape = [int(data.shape().eval()[0])]
        splitData = data.split(0)
        normalizeData = splitData.substract(splitData.mean()).divide(splitData.range())
        return normalizeData.merge(shape)

    def histogram(self, course, ax, sample=None, binSize=0.01):
        idx = featureNames.index(course)
        bins = np.arange(-1, 1+binSize, binSize)
        for name in houseNames:
            data = np.array(self.houseDataByCourse[name][idx].eval())
            if sample is not None:
                si = random.sample(range(len(data)), sample)
                data = data[si]
            ax.hist(data, alpha=0.7, color=houseColor[name], bins=bins)

    def scatterPlot(self, course1, course2, ax, sample=None, size=5):
        idx1 = featureNames.index(course1)
        idx2 = featureNames.index(course2)
        for name in houseNames:
            x = np.array(self.houseDataByCourse[name][idx1].eval())
            y = np.array(self.houseDataByCourse[name][idx2].eval())
            if sample is not None:
                si = random.sample(range(len(x)), sample)
                x = x[si]
                y = y[si]
            ax.scatter(x, y, c=houseColor[name], s=size)

    def pairPlot(self):
        L = len(featureNames)
        for i in tqdm(range(L)):
            for j in range(L):
                ax = plt.subplot(L, L, i*L+j+1)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == j:
                    self.histogram(featureNames[i], ax, sample=50, binSize=0.1)
                else:
                    self.scatterPlot(featureNames[i], featureNames[j], ax, sample=50, size=1)

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
        if np.__version__ >= '1.16':
            data = rf.structured_to_unstructured(rawData[:][featureNames]).view(float)
        else:
            data = rawData[:][featureNames].view(float)
        return np.array(data.reshape(len(rawData), len(featureNames)))

    def readData(self, filename):
        rawData = np.genfromtxt(filename, names=True, delimiter=",", dtype=dataTypes)
        houses = rawData[:][houseColumn]
        data = self.getFeatures(rawData)
        means = np.nanmean(data, axis=0)
        print(means)
        idxs = np.where(np.isnan(data))
        print(idxs)
        data[idxs] = np.take(means, idxs[1])
        return data, houses

    def size(self):
        return len(self.trainData.eval())

    def numberFeatures(self):
        return len(self.trainData.eval()[0])


if __name__ == "__main__":
    data = Data("dataset_train.csv")
    data.describe()
    print(data.dataByStudent[42])
    print(data.dataByStudent.take([42,0,42,1000]))
    print(data.houses["Ravenclaw"])
    data.pairPlot()
    plt.show()
