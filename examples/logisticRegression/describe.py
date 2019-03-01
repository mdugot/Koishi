import argparse

parser = argparse.ArgumentParser(description='Describe a dataset.')
parser.add_argument(
    "dataset",
    type=str,
    help="the csv file containing the data to describe"
)
param = parser.parse_args()

from data import Dataset

dataset = Dataset(param.dataset)
dataset.describe()
