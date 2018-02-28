It appears to CSVIter or something else in MXNet is broken which makes it impossible to train model with CSVIter feeds. I have a reproducible case with CSV MNIST dataset (from here: https://pjreddie.com/projects/mnist-in-csv/)

Original dataset:
mnist_test.csv
mnist_train.csv

Dataset for CVSIter test case:
mnist_iter_train_label.cs
mnist_iter_train_data.csv
mnist_iter_test_label.csv
mnist_iter_test_data.csv

These files are simply derived from original dataset by extracting first column int \*label.csv files
and removing it from \*data.csv files.

1. Download and uncompress cvs files into data/mnist directory https://drive.google.com/file/d/1JaoCKqnFOeR4_FOxijB3b02hImYStnXY/view?usp=sharing
2. Run test case:

working code: mxnet_mnist.py
$ python3 mxnet_mnist.py
Epoch 0. Loss: 0.42287, Train_acc 0.9075, Test_acc 0.9136
Epoch 1. Loss: 0.317125, Train_acc 0.9156, Test_acc 0.9193
Epoch 2. Loss: 0.299069, Train_acc 0.9194, Test_acc 0.9197
Epoch 3. Loss: 0.289855, Train_acc 0.9232, Test_acc 0.9225
...

non working code: mxnet_mnist_iter.py
$ mxnet_mnist_iter.py
Epoch 0. Loss: 0.419388, Train_acc nan, Test_acc 0.9061
Epoch 1. Loss: 0.0, Train_acc nan, Test_acc nan
Epoch 2. Loss: 0.0, Train_acc nan, Test_acc nan
Epoch 3. Loss: 0.0, Train_acc nan, Test_acc nan
Epoch 4. Loss: 0.0, Train_acc nan, Test_acc nan
Epoch 5. Loss: 0.0, Train_acc nan, Test_acc nan
Epoch 6. Loss: 0.0, Train_acc nan, Test_acc nan
Epoch 7. Loss: 0.0, Train_acc nan, Test_acc nan
Epoch 8. Loss: 0.0, Train_acc nan, Test_acc nan
Epoch 9. Loss: 0.0, Train_acc nan, Test_acc nan

