## 1 Tensor 与autograd
![[image/Pasted image 20250319182402.png]]
![[image/Pasted image 20250319182724.png]]
![[image/Pasted image 20250319182734.png]]

## 数据加载
![[image/Pasted image 20250319182759.png]]
![[image/Pasted image 20250319182808.png]]
为了高效地处理数据，PyTorch 提供了强大的工具，包括
`torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader`，
帮助我们管理数据集、批量加载和数据增强等任务

#### 创建数据集(Dataset)
PyTorch 提供了强大的数据加载和处理工具，主要包括：
- **`torch.utils.data.Dataset`**：数据集的抽象类，需要自定义并实现 `__len__`（数据集大小）和 `__getitem__`（按索引获取样本）。
    
- **`torch.utils.data.TensorDataset`**：基于张量的数据集，适合处理数据-标签对，直接支持批处理和迭代。
    
- **`torch.utils.data.DataLoader`**：封装 Dataset 的迭代器，提供批处理、数据打乱、多线程加载等功能，便于数据输入模型训练。
    
- **`torchvision.datasets.ImageFolder`**：从文件夹加载图像数据，每个子文件夹代表一个类别，适用于图像分类任务。

另外，PyTorch 通过 `torchvision.datasets` 模块提供了许多常用的数据集，如**MNIST**
#### 加载数据(DataLoader)


## 神经网络
#### 网络层
PyTorch 提供了一个非常方便的接口来构建神经网络模型，即 `torch.nn.Module`。
我们可以继承 `nn.Module`类并定义自己的网络层，一般需要定义：
```python
class SimpleNN(nn.Module):  
    def __init__(self):  
	    # 定义网络层，直接继承会重写基类的__init__
	    # 要用super ，确保保留父类的参数管理能力
        super(SimpleNN, self).__init__()  

		# 下面是一些网络层
     
    def forward(self, x):  
        # 前向传播过程  
        return x
```
PyTorch 提供了许多常见的神经网络层，以下是几个常见的：
- **`nn.Linear(in_features, out_features)`**：全连接层，输入 `in_features` 个特征，输出 `out_features` 个特征。
- **`nn.Conv2d(in_channels, out_channels, kernel_size)`**：2D 卷积层，用于图像处理。
- **`nn.MaxPool2d(kernel_size)`**：2D 最大池化层，用于降维。
- **`nn.ReLU()`**：ReLU 激活函数，常用于隐藏层。
- **`nn.Softmax(dim)`**：Softmax 激活函数，通常用于输出层，适用于多类分类问题。


#### 激活函数
激活函数决定了神经元是否应该被激活。
它们是非线性函数，使得神经网络能够学习和执行更复杂的任务。
常见的激活函数包括：
- Sigmoid：用于二分类问题，输出值在 0 和 1 之间。
- Tanh：输出值在 -1 和 1 之间，常用于输出层之前。
- ReLU：目前最流行的激活函数之一，定义为 `f(x) = max(0, x)`，有助于解决梯度消失问题。
- Softmax：常用于多分类问题的输出层，将输出转换为概率分布。
```python
import torch.nn.functional as F

# ReLU 激活
output = F.relu(input_tensor)

# Sigmoid 激活
output = torch.sigmoid(input_tensor)

# Tanh 激活
output = torch.tanh(input_tensor)
```

#### 损失函数
损失函数用于衡量模型的预测值与真实值之间的差异。
常见的损失函数包括：
- **均方误差（MSELoss）**：回归问题常用，计算输出与目标值的平方差。
- **交叉熵损失（CrossEntropyLoss）**：分类问题常用，计算输出和真实标签之间的交叉熵。
- **BCEWithLogitsLoss**：二分类问题，结合了 Sigmoid 激活和二元交叉熵损失
```python
# 均方误差损失
criterion = nn.MSELoss()

# 交叉熵损失
criterion = nn.CrossEntropyLoss()

# 二分类交叉熵损失
criterion = nn.BCEWithLogitsLoss()


# 调用示例
loss = criterion(outputs, label)    # 输出值与真实值的误差
```

#### 优化器
优化器负责在训练过程中更新网络的权重和偏置。
常见的优化器包括：
- SGD（随机梯度下降）
- Adam（自适应矩估计）
- RMSprop（均方根传播）
PyTorch 提供了多种优化器，例如 SGD、Adam 等
```python
import torch.optim as optim

# 定义优化器（使用 Adam 优化器）  
optimizer = optim.Adam(model.parameters(), lr=0.001) # lr = 学习率

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降法，学习率和动量  
```

#### 训练
**训练过程：**
1. 前向传播；
2. 计算损失；
3. 反向传播；
4. 更新参数；
```python
model.train()  # 设置模型为训练模式

# 迭代训练  
for epoch in range(100):  # 训练 100 轮  
    output = model(X)  # 前向传播  
    loss = criterion(output, Y)  # 计算损失  
    optimizer.zero_grad()  # 反向传播前要记得清空之前的梯度 
    loss.backward()  # 反向传播  
    optimizer.step()  # 更新参数
```


```python
# 初始化模型、损失函数和优化器  
model = CNN()  
criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失  

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```


dhsfgshf
sfgsgfa
agaga