import koishi
import numpy as np
from tqdm import tqdm

class Model:

    def __init__(self, data, house):
        self.house = house
        self.data = data
        self.initVariable = koishi.uniformInitializer(-0.1,0.1)
        self.feedInputs = koishi.feedInitializer([data.sizeTrainData(), data.numberFeatures()])
        self.feedLabels = koishi.feedInitializer([data.sizeTrainData(),1])
        self.feedTest = koishi.feedInitializer([1, data.numberFeatures()])

        self.inputs = koishi.Variable("inputs", self.feedInputs)
        self.test = koishi.Variable("test", self.feedTest)
        self.labels = koishi.Variable("labels", self.feedLabels)

        self.theta = koishi.Variable([1,data.numberFeatures()], "variable", self.initVariable)
        self.m = koishi.Tensor(data.sizeTrainData())

        self.estimation = self.inputs.matmul(self.theta.transpose([1,0])).sigmoid()
        self.prediction = self.test.matmul(self.theta.transpose([1,0])).sigmoid()

        self.positives = self.estimation.log().multiply(self.labels)
        self.negatives = self.estimation.negative().add(1).log().multiply(self.labels.negative().add(1))
        self.cost = self.m.inverse().negative().multiply(self.positives.add(self.negatives).sum())
        koishi.initializeAll()

    def train(self, epoch, learningRate, optim = None):
        self.feedInputs.feed(np.array(self.data.getTrainData()))
        self.feedLabels.feed(np.array(self.data.getLabels(self.house)))
        print("Train with : " + (optim if optim is not None else "gradient descent"))
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

    def predict(self, features):
        features = [features]
        self.feedTest.feed(features)
        return self.prediction.eval()[0][0]

    def save(self, filename):
        koishi.save(filename, "variable")

    def load(self, filename):
        koishi.load(filename)

class OneVsAll:
    def __init__(self, data):
        self.models = {}
        self.data = data
        for name in data.houses.keys():
            self.models[name] = Model(data, name)

    def train(self):
        for name,model in self.models.items():
            print("train " + name)
            print(model.cost.eval())
            model.train(200, 0.05, optim="adam")
            print(model.cost.eval(), end="\n\n")

    def evaluate(self):
        success = 0
        idx = 0
        print("Evaluate :")
        for features in tqdm(self.data.trainData.eval()):
            result = "unknown"
            prob = 0
            for house,model in self.models.items():
                p = model.predict(features)
                if p > prob:
                    result = house
                    prob = p
            if self.data.labels[result][idx] == 1.:
                success += 1
            idx += 1
        print("%d / %d"%(success,len(self.data.trainData.eval())))

    def resultToCsv(self, filename):
        f = open(filename, "w")
        idx = 0
        f.write("Index,Hogwarts House\n")
        print("Save results to '%s' :" % filename)
        for features in tqdm(self.data.trainData.eval()):
            result = "unknown"
            prob = 0
            for house,model in self.models.items():
                p = model.predict(features)
                if p > prob:
                    result = house
                    prob = p
            f.write("%d,%s\n"%(idx,result))
            idx += 1
        f.close()

if __name__ == "__main__":
    from data import Data
    D = Data("dataset_train.csv")
    predictor = OneVsAll(D)
    predictor.train()
    predictor.evaluate()
    predictor.resultToCsv("/tmp/result.csv")
