import numpy as np
import torch

# nn是神经网络的缩写
from torch import nn
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    # 使用 data.TensorDataset 将输入的数据数组转换成一个 TensorDataset 对象。TensorDataset 是一个将多个张量作为数据集的容器，这些张量应该是相同长度的第一维度。每个数据样本由 TensorDataset 中的张量按顺序组合而成。
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    # 创建一个 DataLoader 实例，它从 TensorDataset 中读取数据，并且能够生成小批量数据。DataLoader 还支持多线程加载以及数据混洗（当 is_train=True 时，默认会启用）。


batch_size = 10
data_iter = load_array((features, labels), batch_size)
# # 创建一个迭代器对象
# iter_data_loader = iter(data_iter)
# for i in range(10):
#     X, y = next(iter_data_loader)
#     print(X, y)


net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 将权重参数初始化为均值为0、标准差为0.01的正态随机数，将偏差参数初始化为零。
# 值得注意的是，我们将两个参数传递到nn.Linear中。 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
loss = nn.MSELoss()
# 损失函数为均方误差损失函数。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 使用小批量随机梯度下降作为优化器，学习率为0.03。
# PyTorch在optim模块中实现了该算法的许多变种。 当我们实例化一个SGD实例时，我们要指定优化的参数 （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。 小批量随机梯度下降只需要设置lr值，这里设置为0.03。
num_epochs = 30
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f"epoch {epoch + 1}, loss {l:f}")

w = net[0].weight.data
print("w的估计误差：", true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print("b的估计误差：", true_b - b)
