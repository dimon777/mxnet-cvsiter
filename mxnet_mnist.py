from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt
from numpy import genfromtxt
mx.random.seed(1)

data_ctx = mx.cpu()
model_ctx = mx.cpu()

def load_data(file, num_features):
    npa = np.genfromtxt(file, delimiter=',')
    npa.astype(np.float32)
    
    X = nd.array(npa[...,1:])
    X = X / 255.0
    Y = nd.array(npa[...,0])

    return X, Y

num_inputs=784
data_shape = (num_inputs,)
label_shape=(1,)
num_outputs = 10

Xtrain, Ytrain = load_data("./data/mnist/mnist_train.csv", num_inputs)
Xtest, Ytest = load_data("./data/mnist/mnist_test.csv", num_inputs)

batch_size = 32
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtrain, Ytrain), batch_size=batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtest, Ytest), batch_size=batch_size, shuffle=False)

net = gluon.nn.Dense(num_outputs)
net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,num_inputs))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

epochs = 10
moving_loss = 0.
num_examples = 60000
loss_sequence = []
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1,num_inputs))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, round(cumulative_loss/num_examples,6), round(train_accuracy,4), round(test_accuracy,4)))
    loss_sequence.append(cumulative_loss)

