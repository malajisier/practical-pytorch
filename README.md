## PyTorch       

官方介绍是一个拥有强力GPU加速的张量和动态构建网络的库，其主要构件是张量，所以我们可以把 PyTorch 当做 NumPy 来用，PyTorch 的很多操作和 NumPy 都是类似的，但是因为其能够在 GPU 上运行，所以有着比 NumPy 快很多倍的速度。

## Chapter 1 - Basic     

### 1. Tensor-basic   

- tensor 类型，默认 float32

- 逐元素操作

  大部分数学函数，激活函数`sigmod,tanh`等，`clamp`函数

- 归并操作

  大多数函数需指定参数`dim`， 在哪个维度上进行操作

- 比较

- 线性代数

  矩阵转置会导致空间不连续，`.contiguous`方法将其转为连续

- tensor 与numpy

  之间具有很高的相似性，转换开销也很小，numpy、tensor共享内存

  

- **key: 广播规则 / broadcast**

  - numpy 广播法则定义：
    - 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分通过在前面加1补齐
    - 两个数组要么在某一个维度的长度一致，要么其中一个为1，否则不能计算
    - 当输入数组的某个维度的长度为1时，计算时沿此维度复制扩充成一样的形状
  - pytorch虽已支持自动广播法则，但一般通过下列函数手动组合实现更为直观、安全：
    - `unsqueeze`或者`view`，或者`tensor[None]`,：为数据某一维的形状补1，实现法则1
    - `expand`或者`expand_as`，重复数组，实现法则3；该操作不会复制数组，所以不会占用额外的空间

  

- tensor内部结构   

  分为头信息区(Tensor)和存储区(Storage)，信息区主要保存着tensor的形状（size）、步长（stride）、数据类型（type）等信息，而真正的数据则保存成连续数组

  

### 2. `tensor`常用操作

- 调整`tensor`形状

  - `view`函数，不会修改自身的数据，返回的新tensor与源tensor共享内存
  - `resize`，新size超过了原size，自动分配新的内存空间；若新size小于原size，则之前的数据依旧会被保存

- 调整维度

  `squeeze`, `unsqueeze`

- 索引操作

### 3. Autograd    

Intro:

`torch.autograd` 自动求导引擎，它能够根据输入和前向传播过程自动构建计算图，并执行反向传播     

PyTorch在autograd模块中实现了计算图的相关功能，autograd中的核心数据结构是Variable。从v0.4版本起，Variable和Tensor合并。我们可以认为需要求导(requires_grad)的tensor即Variable. autograd记录对tensor的操作记录用来构建计算图。



- `requires_grad`

  常用属性：`requires_grad`,`grad_fn`, `is_leaf`

- 计算图 

  autograd会随着用户的操作，记录生成当前variable的所有操作，并由此建立一个有向无环图       

  底层的实现中，图中记录了操作`Function`，每一个变量在图中的位置可通过其`grad_fn`属性在图中的位置推测得到   

  - 设置不自动求导

  - 修改`tensor`的值

  - **查看变量梯度**：

    - hook函数，推荐使用

      输入是梯度，不应返回值，不重复使用需移除

    - `autograd.grad`函数

  - 计算图的特点

### 4. 线性回归与逻辑回归

- 梯度下降算法
- 





## Chapter 2 - Neural Network  

### 1. `torch.nn`    

专门为深度学习而设计的模块。torch.nn的核心数据结构是`Module`，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。
在实际使用中，最常见的做法是继承`nn.Module`，撰写自己的网络/层.

#### 1.1 利用`nn.Module`自定义层

-  自定义层的构造函数`__init__()`需调用`nn.Module`的构造函数
- `Parameter`封装了可学习参数，是一种特殊的`Tensor`，但其默认需要求导
- 定义前向传播函数`forward()`
- 无需写反向传播函数，nn.Module能够利用autograd自动实现反向传播

#### 1.2 多层感知机

- Parameter 参数命名规范

#### 1.3 ` nn.functional`

nn中的大多数layer，在`functional`中都有一个与之相对应的函数 

#### 1.4 `nn.module`深入理解      

- 子Module

- `training`属性 

- hook / 钩子函数

  

### 2. 神经网络层   





