import koishi
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

data, target = load_wine(return_X_y=True)  # load the dataset
data = (data - data.mean(axis=0)) / data.std(axis=0)  # regularize the data
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)  # split between test and train
nfeatures = data.shape[1]

feed_batch = koishi.feedInitializer(list(train_data.shape))
feed_input = koishi.feedInitializer([nfeatures])
init = koishi.uniformInitializer(-0.1,0.1)

inputs = koishi.Variable('inputs', feed_batch)

w1 = koishi.Variable([13, 8], 'w1', init)
w2 = koishi.Variable([8, 3], 'w2', init)

forward = inputs.matmul(w1).matmul(w2)

feed_batch.feed(train_data)
