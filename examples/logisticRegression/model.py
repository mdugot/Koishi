import koishi
import numpy as np
from tqdm import tqdm

class Model:

    def __init__(self, data):
        self.data = data
        self.initVariable = koishi.fillInitializer(0)
        self.feedInputs = koishi.feedInitializer([data.size(), data.numberFeatures()])
        self.feedLabels = koishi.feedInitializer([data.size(),1])

        self.inputs = koishi.Variable("inputs", self.feedInputs)
        self.labels = koishi.Variable("labels", self.feedLabels)

        self.theta = koishi.Variable([1,data.numberFeatures()], "variable", self.initVariable)
        self.m = koishi.Tensor(data.size())

        self.estimation = self.inputs.matmul(self.theta.transpose([1,0]))
#        self.cost = self.estimation.substract(self.outputs).pow(2).sum().divide(self.m.multiply(2))

#        self.feedPrediction = koishi.feedInitializer()
        koishi.initializeAll()

#    def predict(self, km):
#        self.feedPrediction.feed(self.normalize(km, self.kmMin, self.kmRange))
#        return self.unnormalize(self.prediction.eval(), self.priceMin, self.priceRange)

    def train(self, epoch, learningRate, optim = None):
        self.feedInputs.feed(np.array(D.trainData.eval()))
        self.feedOutputs.feed(np.array(self.data.labels.eval()))
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

