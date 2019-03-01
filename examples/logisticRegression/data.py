import koishi
import random
from tqdm import tqdm
import numpy as np
if np.__version__ >= '1.16':
    import numpy.lib.recfunctions as rf
np.warnings.filterwarnings('ignore')
from tqdm import tqdm
import matplotlib.pyplot as plt

dataTypes = ['int','S30','S30','S30','S30','S30','float','float','float','float','float','float','float','float','float','float','float','float','float']
featureNames = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense_Against_the_Dark_Arts', 'Divination', 'Muggle_Studies', 'Ancient_Runes', 'History_of_Magic', 'Transfiguration', 'Potions', 'Care_of_Magical_Creatures', 'Charms', 'Flying']
shortnames = ['Arithmancy', 'Astronomy', 'Herbology', 'DADA', 'Divination', 'MS', 'AR', 'HoM', 'Transf.', 'Potions', 'CMC', 'Charms', 'Flying']
houseColumn = "Hogwarts_House"
houseNames = ['Ravenclaw', 'Hufflepuff', 'Gryffindor', 'Slytherin']
houseColor = {
    "Ravenclaw": 'blue',
    "Hufflepuff": 'gold',
    "Gryffindor": 'red',
    "Slytherin": 'green'}



class Dataset:

    def __init__(self, filename, validation = None, validationPercent = 0.1, shuffle=True):

        data, houses = self.readData(filename)
        np.random.seed(42)
        assert len(data) == len(houses)
        if shuffle is True:
            p = np.random.permutation(len(data))
            data = data[p]
            houses = houses[p]
        if validation is not None:
            limit = int(len(data) * validationPercent)
            if validation is False:
                data = data[:-limit]
                houses = houses[:-limit]
            else:
                data = data[-limit:]
                houses = houses[-limit:]


        self.housesNames = houseNames
        self.numberStudent = len(data)
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

        self.data = self.dataByCourse.transpose([1,0])
        splitData = self.dataByCourse.split(0)
        self.mean = splitData.mean().merge([self.numberFeatures()])
        self.range = splitData.range().merge([self.numberFeatures()])
        self.dataByStudent = self.data.split(0)
        koishi.initializeAll()

        self.houseDataByStudent = {}
        self.houseDataByCourse = {}
        self.labels = {}
        for name in houseNames:
            if len(self.houses[name]) > 0:
                self.houseDataByStudent[name] = self.dataByStudent.take(self.houses[name]).merge([len(self.houses[name])])
                self.houseDataByCourse[name] = self.houseDataByStudent[name].transpose([1,0])
                self.labels[name] = np.zeros([self.numberStudent, 1])
                self.labels[name][self.houses[name],:] = 1.

    def histogram(self, course, ax, sample=None, nbins=100):
        idx = featureNames.index(course)
        data = np.array(self.dataByCourse[idx].eval())
        bins = np.arange(data.min(), data.max(), (data.max()-data.min())/nbins)
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
                ax = plt.subplot(L+1, L+1, (i+1)*(L+1)+j+2)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == j:
                    self.histogram(featureNames[i], ax, sample=50, nbins=20)
                else:
                    self.scatterPlot(featureNames[i], featureNames[j], ax, sample=50, size=1)
                if i == 0 and j == 0:
                    ax.set_title(shortnames[j], loc="right", rotation=-45, pad=5, verticalalignment="bottom")
                elif i == 0:
                    ax.set_title(shortnames[j], rotation="vertical", pad=5, verticalalignment="bottom")
                elif j == 0:
                    ax.set_title(shortnames[i], pad=-15, loc="left", horizontalalignment="right")

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
        idxs = np.where(np.isnan(data))
        data[idxs] = np.take(means, idxs[1])
        return data, houses

    def sizeData(self):
        return len(self.data.eval())

    def getData(self):
        return self.data.eval()

    def getBatch(self, house, idx, batchSize):
        data = self.getData()
        labels = self.getLabels(house)
        assert len(labels) == len(data)
        if (idx+1) * batchSize <= len(data):
            return data[idx*batchSize:(idx+1)*batchSize], labels[idx*batchSize:(idx+1)*batchSize]
        else:
            return None, None

    def getLabels(self, house):
        return self.labels[house]

    def numberFeatures(self):
        return len(self.data.eval()[0])


if __name__ == "__main__":
    data = Data("dataset_train.csv")
    data.describe()
    print(data.dataByStudent[42])
    print(data.dataByStudent.take([42,0,42,1000]))
    print(data.houses["Ravenclaw"])
    data.histogram("Astronomy", plt)
    plt.show()
    data.pairPlot()
    plt.show()
