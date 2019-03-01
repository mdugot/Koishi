import koishi
import numpy as np
import os
from tqdm import tqdm
from data import houseNames

class Model:

    def __init__(self, house, numberFeatures, batchSize):
        self.house = house
        self.batchSize = batchSize
        self.initVariable = koishi.uniformInitializer(-0.1,0.1)
        self.feedInputs = koishi.feedInitializer([1, numberFeatures])
        self.feedBatch = koishi.feedInitializer([batchSize, numberFeatures])
        self.feedLabels = koishi.feedInitializer([batchSize,1])

        self.pp_feedMean = koishi.feedInitializer([numberFeatures])
        self.pp_feedRange = koishi.feedInitializer([numberFeatures])
        self.pp_mean = koishi.Variable(self.getName("preprocess"), self.pp_feedMean)
        self.pp_range = koishi.Variable(self.getName("preprocess"), self.pp_feedRange)

        self.inputs = koishi.Variable(self.getName("inputs"), self.feedInputs)
        self.batch = koishi.Variable(self.getName("batch"), self.feedBatch)
        self.labels = koishi.Variable(self.getName("labels"), self.feedLabels)

        self.ninputs = self.inputs.transpose([1,0]).split(0).substract(
                            self.pp_mean.split(0)).divide(
                                self.pp_range.split(0)).merge(
                                    [numberFeatures]).transpose([1,0])

        self.nbatch = self.batch.transpose([1,0]).split(0).substract(
                            self.pp_mean.split(0)).divide(
                                self.pp_range.split(0)).merge(
                                    [numberFeatures]).transpose([1,0])

        self.theta = koishi.Variable([1,numberFeatures], self.getName("variable"), self.initVariable)
        self.m = koishi.Tensor(batchSize)

        self.estimation = self.ninputs.matmul(self.theta.transpose([1,0])).sigmoid()
        self.outputs = self.nbatch.matmul(self.theta.transpose([1,0])).sigmoid()

        self.positives = self.outputs.log().multiply(self.labels)
        self.negatives = self.outputs.negative().add(1).log().multiply(self.labels.negative().add(1))
        self.cost = self.m.inverse().negative().multiply(self.positives.add(self.negatives).sum())
        koishi.initializeAll()

    def getName(self, name):
        return "%s/%s"%(self.house,name)

    def optim(self, learningRate, optim):
        if optim == "momentum":
            koishi.momentumOptim(self.getName("variable"), learningRate, 0.9)
        elif optim == "rmsprop":
            koishi.RMSPropOptim(self.getName("variable"), learningRate, 0.9)
        elif optim == "adam":
            koishi.adamOptim(self.getName("variable"), learningRate, 0.9, 0.9)
        else:
            koishi.gradientDescentOptim(self.getName("variable"), learningRate)

    def train(self, trainset, epoch, learningRate, optim = None, stochastic=False):
        self.pp_feedMean.feed(trainset.mean.eval())
        self.pp_feedRange.feed(trainset.range.eval())
        print("Train with : " + (optim if optim is not None else "gradient descent"))
        for i in tqdm(range(epoch)):
            idx = 0
            features,labels = trainset.getBatch(self.house, idx, self.batchSize)
            while features is not None:
                self.feedBatch.feed(features)
                self.feedLabels.feed(labels)
                self.cost.backpropagation()
                if stochastic is True:
                    self.optim(learningRate, optim)
                features,labels = trainset.getBatch(self.house, idx, self.batchSize)
                idx += 1
            if stochastic is False:
                self.optim(learningRate, optim)

    def predict(self, features):
        features = [features]
        self.feedInputs.feed(features)
        return self.estimation.eval()[0][0]

    def save(self):
        path = "./saves/"
        if not os.path.exists(path):
            os.makedirs(path)
        koishi.save(os.path.join(path, self.house), [self.getName("preprocess"), self.getName("variable")])

    def load(self):
        path = os.path.join("./saves/", self.house)
        if not os.path.exists(path):
            print("save not found")
        koishi.load(path)

class OneVsAll:
    def __init__(self, numberFeatures, batchSize):
        self.models = {}
        for name in houseNames:
            self.models[name] = Model(name, numberFeatures, batchSize)

    def train(self, dataset):
        for name,model in self.models.items():
            print("train " + name)
            model.train(dataset, 20, 0.1, optim="adam", stochastic=False)

    def evaluate(self, dataset):
        success = 0
        idx = 0
        print("Evaluate :")
        for features in tqdm(dataset.getData()):
            result = "unknown"
            prob = 0
            for house,model in self.models.items():
                p = model.predict(features)
                if p > prob:
                    result = house
                    prob = p
            if result in dataset.labels.keys() and dataset.labels[result][idx] == 1.:
                success += 1
            idx += 1
        print("%d / %d"%(success,len(dataset.getData())))

    def resultToCsv(self, filename, dataset):
        f = open(filename, "w")
        idx = 0
        f.write("Index,Hogwarts House\n")
        print("Save results to '%s' :" % filename)
        for features in tqdm(dataset.getData()):
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

    def save(self):
        for name,model in self.models.items():
            model.save()

    def load(self):
        for name,model in self.models.items():
            model.load()

if __name__ == "__main__":
    from data import Data
    trainset = Data("dataset_train.csv", validation=False)
    testset = Data("dataset_train.csv", validation=True)
    assert trainset.numberFeatures() == testset.numberFeatures()
    model = OneVsAll(trainset.numberFeatures(), 400)
    model.train(trainset)
    model.evaluate(trainset)
    model.evaluate(testset)
    model.save()
