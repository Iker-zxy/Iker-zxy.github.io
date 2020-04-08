---
title: Loss Function
categories:
- 机器学习
tags:
- Loss
---

### Softmax交叉熵损失函数求导公式

对于多分类问题，我们一般使用Softmax函数作为输出层的激活函数，用交叉熵作为损失函数。

Softmax函数因其公式中分母为所有项之和，求导时非常不便，故特意单独拎出来进行推导，令
$$
z_{i}=\sum_{j}w_{ij}x_{ij}+b\\
y_{i}=\sigma(z_{i})=\frac{e^{z_{i}}}{\sum_{k}e^{z_{k}}}\\
$$
损失函数为
$$
J=-\sum_{i}t_{i}\ln y_{i}
$$
**推导过程:** 

根据链式法则，有
$$
\frac{\partial J}{\partial z_{i}}=\sum_{j}\frac{\partial J}{\partial y_{j}}\frac{\partial y_{j}}{\partial z_{i}}
$$
上述的公式中会有求和的形式，主要是由于softmax公式的特性，它的分母包含了所有神经元的输出，所以，对于不等于$i$的其他输出里面，也包含着$z_{i}$，所有的$y$都要纳入到计算范围中
$$
\frac{\partial J}{\partial y_{j}}=\frac{\partial(-\sum_{i}t_{i}\ln y_{i})}{\partial y_{j}}=-\frac{t_{j}}{y_{j}}
$$
对于$\frac{\partial y_{j}}{\partial z_{i}}$,需要分为$i=j$和$i≠j$两种情况求导

1. 若$j = i$
   $$
   \frac{\partial y_{i}}{\partial z_{i}}=\frac{\partial \frac{e^{z_{i}}}{\sum_{k}e^{z_{k}}}}{\partial z_{i}} =\frac{e^{z_{i}}\sum_{k}e^{z_{k}}-(e^{z_{i}})^{2}}{(\sum_{k}e^{z_{k}})^{2}}=(\frac{e^{z_{i}}}{\sum_{k}e^{z_{k}}})(1-\frac{e^{z_{i}}}{\sum_{k}e^{z_{k}}})=y_{i}(1-y_{i})
   $$

2. 若$j\neq i$

$$
\frac{\partial y_{i}}{\partial z_{j}}=\frac{\partial \frac{e^{z_{j}}}{\sum_{k}e^{z_{k}}}}{\partial z_{i}} =-\frac{e^{z_{j}}e^{z_{i}}}{(\sum_{k}e^{z_{k}})^{2}}=-y_{i}y_{j}
$$

把两部分相结合，可以得到
$$
\frac{\partial J}{\partial z_{i}}=\sum_{j}(-\frac{t_{j}}{y_{j}})\frac{\partial y_{j}}{\partial z_{i}}=-\frac{t_{i}}{y_{i}}y_{i}(1-y_{i})+\sum_{j\neq i}\frac{t_{j}}{y_{j}}y_{i}y_{j}=-t_{i}+t_{i}y_{i}+\sum_{j\neq i}t_{j}y_{i}=-t_{i}+y_{i}\sum_{j}t_{j}
$$
对于分类问题，只有一个类别为1，其余都为0，所以$\sum_{j}t_{j}=1$

所以对于分类问题，有
$$
\frac{\partial J}{\partial z_{i}}=y_{i}-t_{i}
$$

###