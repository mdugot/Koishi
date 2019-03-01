import argparse

parser = argparse.ArgumentParser(description='Create and train logistic regression model.')
parser.add_argument(
    "dataset",
    type=str,
    help="the csv file containing the data to use for training"
)
param = parser.parse_args()

from data import Dataset
from model import OneVsAll

trainset = Dataset(param.dataset, validation=False)
testset = Dataset(param.dataset, validation=True)
assert trainset.numberFeatures() == testset.numberFeatures()
model = OneVsAll(trainset.numberFeatures(), trainset.sizeData()//5)
model.train(trainset)
model.evaluate(trainset)
model.evaluate(testset)
model.save()

