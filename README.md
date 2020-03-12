## PyTorch       

官方介绍是一个拥有强力GPU加速的张量和动态构建网络的库，其主要构件是张量，所以我们可以把 PyTorch 当做 NumPy 来用，PyTorch 的很多操作和 NumPy 都是类似的，但是因为其能够在 GPU 上运行，所以有着比 NumPy 快很多倍的速度。

## Chapter 1 - Basic     

#### 1.自动扩展 / broadcast

  - numpy 广播法则定义：
    - 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分通过在前面加1补齐
    - 两个数组要么在某一个维度的长度一致，要么其中一个为1，否则不能计算
    - 当输入数组的某个维度的长度为1时，计算时沿此维度复制扩充成一样的形状
  - pytorch虽已支持自动广播法则，但一般通过下列函数手动组合实现更为直观、安全：
    - `unsqueeze`或者`view`，或者`tensor[None]`,：为数据某一维的形状补1，实现法则1
    - `expand`或者`expand_as`，重复数组，实现法则3；该操作不会复制数组，所以不会占用额外的空间


#### 2.Tensor运算    
（1）squeeze / unsqueeze     
（2）相乘    
- torch.mm
  只适用于2d的tensor
- torch.matmul
  支持多维，推荐使用
- @
  相当于matmul，matmul的重载    

（3）高阶操作
- where
  torch.where(condition, x, y)
- gather
  


#### 3.Autograd/自动求导    


`torch.autograd` 自动求导引擎，它能够根据输入和前向传播过程自动构建计算图，并执行反向传播     

PyTorch在autograd模块中实现了计算图的相关功能，autograd中的核心数据结构是Variable。从v0.4版本起，Variable和Tensor合并。我们可以认为需要求导(requires_grad)的tensor即Variable. autograd记录对tensor的操作记录用来构建计算图。


#### 计算图 
autograd会随着用户的操作，记录生成当前variable的所有操作，并由此建立一个有向无环图       
底层的实现中，图中记录了操作`Function`，每一个变量在图中的位置可通过其`grad_fn`属性在图中的位置推测得到   

- 





## Chapter 2 - Neural Network  

### `torch.nn`    

专门为深度学习而设计的模块。torch.nn的核心数据结构是`Module`，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。
在实际使用中，最常见的做法是继承`nn.Module`，撰写自己的网络/层.

  

### 2. 神经网络层   





## Optional     


### keys       

#### 1. `torch.backends.cudnn.benchmark = True`

**解决办法**

总的来说，大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。

**一般来讲，应该遵循以下准则：**

- 如果网络的**输入数据维度或类型上变化不大**，设置 `torch.backends.cudnn.benchmark = true`
  可以增加运行效率；
- 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。

