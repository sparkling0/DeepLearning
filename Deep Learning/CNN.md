上一次采用**全连接前馈网络**进行特征提取，将一张图片上的所有像素点展开成一个 1 维向量输入网络，这种方式存在如下两个问题：
1. 模型**参数过多**，容易发生过拟合。在全连接网络中，隐藏层的每个神经元都要跟该层所有输入的神经元相连接，都会有一个权重。随着隐藏层神经元数量的增多，参数的规模也会急剧增加，导致整个神经网络的训练效率非常低，也很容易发生过拟合。
2. 难以提取图像中的**局部不变性**特征。自然图像中的物体都具有局部不变性特征，比如尺度缩放、平移、旋转等操作不影响其语义信息。而全连接前馈网络很难提取这些局部不变性特征。

为了解决上述问题，引入了**卷积神经网络**(CNN)进行特征提取，卷积神经网络在标准的前馈神经网络的基础上加了一些**卷积层**和**池化层**。
对于输入，不再需将图片展开成一维向量，而是用一个**卷积核**对整张图片进行特征提取，这时输入层到卷积层间的卷积核，就替代了$w$，因此模型需要不断训练，更新CNN中的卷积和（和全连接层中的$w，b$）
一个简单的卷积神经网络如下图所示：![[image/Pasted image 20250312205105.png]]

## 1 卷积层
#### 1.1 什么是卷积
卷积运算用一个称为**卷积核**（滤波器）的**矩阵**从上到下、从左到右在图像上滑动，将卷积核矩阵的各个元素与它在图像上覆盖的对应位置的元素相乘，然后求和，得到输出值，称为特征图（Feature Map）。
	![[image/Pasted image 20250312204322.png]]

这样就能以**一块区域**为单位提取特征，不同与FNN中只以一个单一的值。

 卷积核每次滑动的大小称为**步幅（stride）**，上面的例子中卷积核每次滑动一个像素点，步幅为 1。下图为步幅为 2 的卷积过程![[image/Pasted image 20250312204849.png]]
 

不同卷积运算使用卷积核矩阵遍历整个图像，对所有位置进行卷积操作，得到输出图像，经过卷积运算之后图像尺寸变小了，完成了对输入图像的降维和特征提取。
如果原始图像是 $m×n$，卷积核为 $s×s$，则卷积结果图像的尺寸为:
		$$(m-s+1)×(n-s+1)$$
当卷积核尺寸大于 1 时，输出特征图的尺寸会小于输入图片尺寸。
在深层网络中，经过多次卷积，输出图片尺寸会不断减小，而对于一张图片的边缘部分，其像素在卷积时参与计算的次数较少（例如角落像素仅被卷积核覆盖一次），因此，如果边界和角落像素丢失越来越多，导致模型训练效果不佳。
为了避免特征图过早缩小，保留更多空间信息，更充分地提取边缘特征，往往会对卷积核进行**填充（Padding）**，如下：![[image/Pasted image 20250312204807.png]]
且当卷积核的高度和宽度不同时，也可以通过填充使输入和输出具有相同的宽和高。



#### 1.2 作用
在数字图像处理领域，卷积是一种常见的运算。它可以用于图像去噪、增强、边缘检测等问题，还可以用于提取图像的特征。
正如上面说过，它是以一块区域为单位，能提取图像的**局部特性**。
例如在边缘检测中，有专门的卷积核矩阵，如Sobel边缘检测算子为例，它的卷积和矩阵为：
	$$\begin{bmatrix}
	-1&-1&-1\\
	0&0&0\\
	1&2&1\\
	\end{bmatrix}$$
对于输入一张图片，通过将卷积核从上到下、从左到右依次作用于输入图像的所有位置，可以得到图像的边缘图。边缘图在边缘位置有更大的值，在非边缘处的值接近于0。
除 Sobel 算子之外，常用的还有 Roberts 算子、Prewitt 算子等，它们实现卷积的方法相同，但使用了不同的卷积核矩阵。通过这种卷积运算可以抽取图像的边缘，如果使用其他卷积核，可以抽取更一般的图像特征。
在图像处理中，这些卷积核矩阵的数值是根据经验**人工设计**的，也可以通过**机器学习**的手段来自动生成这些卷积核。
卷积神经网络是通过**自学习**的手段来得到各种有用的卷积核，即卷积核就是在CNN中我们需要更新的参数

#### 1.3 说明
每个卷积核是一个特征提取器，图像中所有位置处的卷积操作共享这个卷积核的权重。

假设输入图像的子图像在$(i , j)$位置的像素值为$x_{ij}$，卷积核矩阵在位置$(p , q)$的元素值为$k_{pq}$。卷积核作用于图像的某一位置，得到的输出为：
$$
f(\sum\limits_{p=1}^{s}\sum\limits_{q=1}^{s}k_{pq}x_{i+p-1,j+p-1}+b)
$$

其中，与FNN中一样，f 为激活函数，b 为偏置项，保证非线性。

与FNN中的$W$一样，每一层需要多个卷积核，抽取各种不同的特征。

一般都需要设计多个卷积层。因为我们要在不同的尺度和层次上进行特征抽取，如果只有一个卷积层，就只能处理一个尺度。
对于不同层，卷积核在一次卷积操作时对原图像的作用范围称为**感受野**，指的是神经网络中神经元“看到的”输入区域，不同的卷积层有不同的感受野。网络前面的卷积层感受野小，用于提取图像细节的信息；后面的卷积层感受野更大，用于提取更大范围的、高层的抽象信息，这是多层卷积网络的设计初衷。
如下图所示，`kernel_size = 3, stride = 1`，
Layer2 中的绿色区域表示了Layer1的绿色区域，
Layer3 中的黄色区域表示了Layer2的黄色区域，进一步表示了Layer1的黄色区域
最后的输出结果也就代表了整张原始像的特征

![[image/Pasted image 20250313204942.png]]

另外，前面讲述的是**单通道**图像卷积，即输入是二维数组。实际应用时遇到的经常是**多通道**图像，如 RGB 彩色图像有三个通道，此时用三个通道的卷积核的各个通道**分别**对输入图像的各个通道进行卷积，然后把对应位置处的像素值按照各个通道**累加**：![[image/Pasted image 20250313200204.png]]

其次，由于每一层可以有多个卷积核，产生的输出也是多通道的特征图像，此时对应的卷积核也是多通道的。由于每一层允许有多个卷积核，卷积操作后会输出多张特征图像，因此，第l 个卷积层每个**卷积核的通道数必须与输入特征图像的通道数相同**，即第 $l$ 层的卷积核通道数等于第 $l-1$ 层卷积核的个数。通常将卷积核的输出特征图的通道数叫做卷积核的个数。
![[image/Pasted image 20250313200356.png]]

#### 1.4 Pytorch
使用 `nn.Module` 构建一个 CNN
```python
# 第一层卷积层，输入1通道，输出32通道，卷积核大小3x3  
self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  

# 第一层卷积层，输入32通道，输出64通道，卷积核大小3x3  
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
```


	
## 2 池化层
通过卷积操作，完成了对输入图像的降维和特征提取，但特征图像的维数还是很高。维数高不仅计算耗时，而且容易导致过拟合。为此，引入了下采样技术，也称为**池化**（Pooling）操作。如下图所示为两种常用的池化操作。

#### 2.1 什么是池化
最基本的池化操作的做法是对图像的某一个区域用一个值代替，如最大值或平均值。
如果采用最大值，称为最大池化；如果采用平均值，称为均值池化。
![[image/Pasted image 20250313200652.png]]
除了降低图像尺寸之外，池化带来的另一个好处是一定程度的平移、旋转不变性，因为输出值由图像的一片区域计算得到，对于小幅度的平移和旋转不敏感。
如下图所示，左边是一个标准的 x 图像，右边是一个被旋转了的 x 图像，如果我们使用最大池化或均值池化操作，黄色框里边池化后的值是不变的，因此，其对小幅度的旋转是不敏感的。           
![[image/Pasted image 20250313200822.png]]
池化层实现时是在进行卷积操作之后对得到的特征图像进行**分块**，图像被划分成不相交的块，计算这些块内的最大值或平均值，得到池化后的图像。
均值池化和最大池化都可以完成降维操作，一般情况下最大池化有更好的效果。

#### 2.2 Pytorch
```python
x = torch.nn.functional.relu(self.conv1(x))  # 第一层卷积 + ReLU
x = torch.nn.functional.max_pool2d(x, 2)     # 最大池化
```


## 3 全连接层
一张图片在经过卷积层和池化层后，得到一张或多张池化后的图像，接下来就是上次的FNN模型。

现将图片展开成一维向量，依次经过输入层，隐藏层，输出层，卷积神经网络的全连接层和我们前面讲的全连接神经网络相同。
**展开代码：**
```python
x = F.max_pool2d(x, 2)       # 最后一次最大池化
x = x.view(-1, 64 * 7 * 7) # 展平操作
```


## 4 模型训练
#### 4.1 训练步骤
按照一般的神经网络训练步骤训练：
1. **准备数据；**
2. **定义损失函数和优化器；**
3. **开始训练**：
	1. *前向传播；*
	2. *计算损失；*
	3. *反向传播；*
    4. *更新参数；*


其中，CNN的正向传播算法与全连接神经网络类似，只不过输入的是二维或者更高维的图像，输入数据依次经过每个层，最后产生输出。卷积层、池化层的正向传播计算方法就是前面两小节卷积计算和池化操作的过程，再结合全连接层的正向传播方法，可以得到整个卷积神经网络的正向传播算法。
在全连接神经网络中，权重和偏置通过反向传播算法训练得到，卷积网络的训练同样使用反向传播算法。推导还是比较复杂的，但是用Pytorch实际直接调用还是很简单，与FNN差别不大

#### 4.2 Pytorch
```python
# 定义损失函数和优化器
model = CNN()  
criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失  
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降法，学习率和动量  
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
  
# 训练模型  
num_epochs = 5  
model.train()   # 设为训练模式  
for epoch in range(num_epochs):  
    total_loss = 0  
	for image, label in train_loader:  
        # 前向传播  
        outputs = model(image)  
        loss = criterion(outputs, label)  
  
        # 反向传播  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
        total_loss += loss.item()
```





## 5 PyTorch 实现一个 CNN 实例
以下示例展示如何用 PyTorch 构建一个简单的 CNN 模型，用于 MNIST 数据集的数字分类。
主要步骤
```python
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
```

1. **数据加载**
使用 `torchvision` 加载和预处理 MNIST 数据。
```python
train_dataset = MNIST(root='./', train=True, transform=transforms.ToTensor(),download=False)  
test_dataset = MNIST(root='./', train=False, transform=transforms.ToTensor(), download=False)  
  
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```
2. 网络设计
```python
# 网络构建  
class CNN(nn.Module):  
    def __init__(self):  
        super(CNN, self).__init__()  
        # 卷积层  
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)    # 第一个卷积层  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5, padding=2)  
  
        # 全连接层  
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)   # 输入大小 = 特征图大小 * 通道数  
        self.fc2 = nn.Linear(in_features=128, out_features=10)  
  
    def forward(self, x):  
        x = F.relu(self.conv1(x))            # 第一层卷积 + ReLU        
        x = F.max_pool2d(x, 2)     # 最大池化  
        x = F.relu(self.conv2(x))            # 第二层卷积 + ReLU        
        x = F.max_pool2d(x, 2)     # 最大池化  
        x = x.view(-1, 64 * 7 * 7)           # 展平操作  
        x = F.relu(self.fc1(x))              # 全连接层 + ReLU        
        x = self.fc2(x)                      # 全连接层输出  
  
        return x
```
2. 训练配置
```python
# 训练配置  
# 初始化模型、损失函数和优化器  
model = CNN()  
criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失  
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降法，学习率和动量  
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```
2. 开始训练
	1. a
	2. b
	3. c
3. 评估