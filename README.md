## Contents:         

### PyTorch       

官方介绍是一个拥有强力GPU加速的张量和动态构建网络的库，其主要构件是张量，所以我们可以把 PyTorch 当做 NumPy 来用，PyTorch 的很多操作好 NumPy 都是类似的，但是因为其能够在 GPU 上运行，所以有着比 NumPy 快很多倍的速度。



### Chapter 1 - Basic     

1. Tensor-basic   
   
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

### Chapter 2 - Neural Network  

