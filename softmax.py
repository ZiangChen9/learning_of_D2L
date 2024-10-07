import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(X):
    return softmax(torch.matmul(X.reshape((-1, num_inputs)), w) + b)


def cross_entropy(y_hat, y):
    # 定义交叉熵
    print(type(-torch.log(y_hat[range(len(y_hat)), y])))
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    # y_hat是矩阵且矩阵的列数大于1
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 使用 argmax 函数沿着指定的轴（这里是 axis=1）找到最大值的索引
        y_hat = y_hat.argmax(axis=1)
    # 类型转换：y_hat.type(y.dtype)将y_hat张量的数据类型转换为与y张量的数据类型相同。
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
