import koishi
from tqdm import tqdm
import matplotlib.pyplot as plt

class lr:

    def __init__(self, filename):
        self.km,self.price = self.readData(filename)
        self.initVariable = koishi.fillInitializer(0)
        self.initSize = koishi.fillInitializer(len(self.km))
        self.feedInputs = koishi.feedInitializer(self.km)
        self.feedOutputs = koishi.feedInitializer(self.price)
        
        self.inputs = koishi.Tensor([len(self.km),1], "inputs", self.feedInputs)
        self.outputs = koishi.Tensor([len(self.km),1], "outputs", self.feedOutputs)
        self.theta0 = koishi.Tensor("variable", self.initVariable)
        self.theta1 = koishi.Tensor([1,1], "variable", self.initVariable)
        self.m = koishi.Tensor("size", self.initSize)
        self.estimation = self.inputs.matmul(self.theta1).add(self.theta0)
        self.cost = self.estimation.substract(self.outputs).pow(2).sum().divide(self.m.multiply(2))

    
        self.feedPrediction = koishi.feedInitializer(0)
        self.oneInput = koishi.Tensor("oneInput", self.feedPrediction)
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
        self.feedPrediction.init()
        return self.unnormalize(self.prediction.eval(), self.priceMin, self.priceRange)

    def plotResult(self, toPredict = None):
        plt.plot(self.rawKm, self.rawPrice, 'bo')
        kmMin = self.kmMin
        kmMax = max(self.rawKm)
        if toPredict < kmMin:
            kmMin = toPredict
        if toPredict > kmMax:
            kmMax = toPredict
        a = self.predict(kmMin)
        b = self.predict(kmMax)
        plt.plot([kmMin, kmMax], [a,b], 'r-')
        if toPredict is not None:
            p = self.predict(toPredict)
            plt.plot([toPredict], [p], 'gx')
        plt.show()
        
    
    def train(self, epoch, learningRate):
        for i in tqdm(range(epoch)):
            self.cost.gradientReinit()
            self.cost.gradientUpdate()
            koishi.gradientDescent("variable", learningRate)

    def save(self, filename):
        koishi.save(filename, "variable")

    def load(self, filename):
        koishi.load(filename)
