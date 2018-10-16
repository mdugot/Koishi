import koishi
from tqdm import tqdm
import matplotlib.pyplot as plt

class lr:

    def __init__(self, filename):
        self.km,self.price = self.readData(filename)
        self.initVariable = koishi.fillInitializer(0)
        self.feedInputs = koishi.feedInitializer()
        self.feedOutputs = koishi.feedInitializer()
        
        self.inputs = koishi.Variable([len(self.km),1], "inputs", self.feedInputs)
        self.outputs = koishi.Variable([len(self.km),1], "outputs", self.feedOutputs)
        self.theta0 = koishi.Variable("variable", self.initVariable)
        self.theta1 = koishi.Variable([1,1], "variable", self.initVariable)
        self.m = koishi.Tensor(len(self.km))
        self.estimation = self.inputs.matmul(self.theta1).add(self.theta0)
        self.cost = self.estimation.substract(self.outputs).pow(2).sum().divide(self.m.multiply(2))

    
        self.feedPrediction = koishi.feedInitializer()
        self.oneInput = koishi.Variable("oneInput", self.feedPrediction)
        self.prediction = self.oneInput.multiply(self.theta1[0][0]).add(self.theta0)

        koishi.initializeAll()

    def normalize(self, value, minv, rangev):
        return (value-minv) / rangev

    def unnormalize(self, value, minv, rangev):
        return value*rangev + minv

    def readData(self, filename):
        with open(filename) as f:
            lines = f.readlines()
        lines = [l.strip("\n").split(",") for l in lines[1:]]
        km = [float(l[0]) for l in lines]
        price = [float(l[1]) for l in lines]
        self.kmRange = max(km) - min(km)
        self.priceRange = max(price) - min(price)
        self.kmMin = min(km)
        self.priceMin = min(price)
        self.rawKm = km
        self.rawPrice = price
        km = [self.normalize(v, self.kmMin, self.kmRange) for v in km]
        price = [self.normalize(v, self.priceMin, self.priceRange) for v in price]
        return km, price

    def predict(self, km):
        self.feedPrediction.feed(self.normalize(km, self.kmMin, self.kmRange))
        return self.unnormalize(self.prediction.eval(), self.priceMin, self.priceRange)

    def plotResult(self, toPredict = None):
        plt.plot(self.rawKm, self.rawPrice, 'bo')
        kmMin = self.kmMin
        kmMax = max(self.rawKm)
        if toPredict is not None and toPredict < kmMin:
            kmMin = toPredict
        if toPredict is not None and toPredict > kmMax:
            kmMax = toPredict
        a = self.predict(kmMin)
        b = self.predict(kmMax)
        plt.plot([kmMin, kmMax], [a,b], 'r-')
        if toPredict is not None:
            p = self.predict(toPredict)
            plt.plot([toPredict], [p], 'gx')
        plt.show()
        
    
    def train(self, epoch, learningRate, optim = None):
        self.feedInputs.feed(self.km)
        self.feedOutputs.feed(self.price)
        if optim != None:
            print("Train with : " + optim)
        for i in tqdm(range(epoch)):
            self.cost.backpropagation()
            if optim == "momentum":
                koishi.momentumOptim("variable", learningRate, 0.9)
            elif optim == "rmsprop":
                koishi.RMSPropOptim("variable", learningRate, 0.9)
            elif optim == "adam":
                koishi.adamOptim("variable", learningRate, 0.9, 0.9)
            else:
                koishi.gradientDescentOptim("variable", learningRate)

    def save(self, filename):
        koishi.save(filename, "variable")

    def load(self, filename):
        koishi.load(filename)
