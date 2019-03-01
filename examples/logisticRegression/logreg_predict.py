import argparse

parser = argparse.ArgumentParser(description='Use a trained logistic regression model for prediction.')
parser.add_argument(
    "dataset",
    type=str,
    help="the csv file containing the data to use for prediction"
)
param = parser.parse_args()


from data import Dataset
from model import OneVsAll

dataset = Dataset(param.dataset, shuffle=False)
model = OneVsAll(dataset.numberFeatures(), 1)
model.load()
model.resultToCsv("./houses.csv", dataset)
