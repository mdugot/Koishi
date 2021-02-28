import koishi
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data, target = load_iris(return_X_y=True)  # load the dataset
data = (data - data.mean(axis=0)) / data.std(axis=0)  # regularize the data
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)  # split between test and train
nfeatures = data.shape[1]
nclasses = target.shape[0]

batch_size = 10
epoch = 10
epoch = 10
learning_rate = 0.01

feed_inputs = koishi.feedInitializer([batch_size, nfeatures])
feed_targets = koishi.feedInitializer([batch_size])
init = koishi.uniformInitializer(-0.1,0.1)

inputs = koishi.Variable('inputs', feed_inputs)
targets = koishi.Variable('targets', feed_targets)

w1 = koishi.Variable([4, 4], 'param', init)
w2 = koishi.Variable([4, 3], 'param', init)
b1 = koishi.Variable([4], 'param', init)
b2 = koishi.Variable([3], 'param', init)

forward = inputs.matmul(w1).add(b1).sigmoid().matmul(w2).add(b2).softmax(1)
loss = forward.get(targets).log().negative().mean()


koishi.initializeAll()
feed_inputs.feed(train_data[:batch_size])
feed_targets.feed(train_target[:batch_size])

print("Inputs : ", inputs.eval())
print("Targets : ", targets.eval())
print("Forward : ", forward.eval())
print("Loss : ", loss.eval())
