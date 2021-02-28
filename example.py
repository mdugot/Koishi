import koishi
import numpy as np
import pandas
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print('downloading the dataset...')
data, target = fetch_kddcup99(return_X_y=True, as_frame=True, percent10=True, shuffle=True)
print('replace string features by one hot vector...')
data = pandas.get_dummies(data, columns=['protocol_type', 'service', 'flag'])
print('replace labels by integer...')
labels = {label for label in target}
labels_map = dict(zip(labels, range(len(labels))))
target = target.map(labels_map)
print('regularize...')
data = (data - data.mean(axis=0)) / (data.std(axis=0) + 0.000001)
data = data.to_numpy()
target = target.to_numpy()
print('data ready !')

nfeatures = int(data.shape[1])
nclasses = int(target.max() + 1)
hidden_layer = 128

batch_size = 100

epoch = 1

learning_rate = 0.0002
momentum = 0.9
rms = 0.9

print("make initializer...")
feed_inputs = koishi.feedInitializer([batch_size, nfeatures])
feed_labels = koishi.feedInitializer([batch_size])
init = koishi.uniformInitializer(-0.1,0.1)

print("create model parameters...")
inputs = koishi.Variable('inputs', feed_inputs)
labels = koishi.Variable('labels', feed_labels)

w1 = koishi.Variable([nfeatures, hidden_layer], 'param', init)
w2 = koishi.Variable([hidden_layer, nclasses], 'param', init)
b1 = koishi.Variable([hidden_layer], 'param', init)
b2 = koishi.Variable([nclasses], 'param', init)

print("create model and loss function...")
forward = inputs.matmul(w1).add(b1).sigmoid().matmul(w2).add(b2).softmax(1)
loss = forward.get(labels).log().negative().mean()


print("start training...")
koishi.initializeAll()

loss_history = []
acc_history = []
try:
    for batch_idx in range(len(data) // batch_size):
        start = batch_idx*batch_size
        end = (batch_idx + 1)*batch_size
        feed_inputs.feed(data[start:end])
        feed_labels.feed(target[start:end])
        predictions = np.argmax(forward.eval(), axis=1)
        true_labels = labels.eval()
        running_loss = loss.mean().eval()
        running_acc = (predictions == true_labels).sum() / batch_size
        print(f"[{batch_idx*batch_size}/{len(data)}] Loss : {running_loss:.4}, Accuracy : {running_acc:.2}       ", end='\r')
        loss_history.append(running_loss)
        acc_history.append(running_acc)
        loss.backpropagation()
        koishi.adamOptim(learning_rate, momentum, rms)
except KeyboardInterrupt:
    pass

ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

ax1.plot(loss_history)
ax1.set_title("Loss")
ax1.set_xlabel("steps")
ax1.set_ylabel("loss")

ax2.plot(acc_history)
ax2.set_title("Accuracy")
ax2.set_xlabel("steps")
ax2.set_ylabel("accuracy")

plt.show()
