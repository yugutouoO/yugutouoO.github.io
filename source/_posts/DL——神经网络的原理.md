---
title: DL——神经网络的原理
date: 2019-05-30 11:18:24
tags: 
- DL
- 梯度下降法
top:
categories: DL
---
# 数学

## 导数

## 方向导数

方向导数是函数在某个（方向）坐标轴上的变化率。

## 梯度

梯度是一个向量，方向是方向导数最大（最陡峭）的方向。

# 梯度下降法

> 如你在山顶的一点，往下看，有很多不同的方向，不同的方向有不同的陡峭程度。如果你想最快下山，就每次朝着最陡峭（梯度方向）的方向走一小步。

# 神经网络

m个样本，每个样本的x有n个特征

+  <font color=red>真实值 vs 估计值</font>

样本：$[x^{(i)}_1,x^{(i)}_2,…,x^{(i)}_m]$	$y^{(i)}$

样本估计：$z_{\theta}(x^{(i)})=\theta_1x^{(i)}_1+\theta_2x^{(i)}_2+…+\theta_nx^{(i)}_n$

（一条样本的）样本估计和真实值的loss：

$J_{(\theta)}=\frac12\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$



目的是获得最小化样本估计和真实值之间的误差的参数$\theta$（n维）

+  <font color=red>方法：梯度下降。</font>

$\frac{\partial}{\partial\theta_j}J(\theta)=\frac{\partial}{\theta_j}\frac12(h_{\theta}(x)-y)^2$

$=2\cdot\frac12(h_{\theta}(x)-y)\cdot \frac{\partial}{\theta_j}(h_{\theta}(x)-y)$

$=(h_{\theta}(x)-y)\cdot \frac{\partial}{\partial{\theta_j}}(\sum_{i=0}^{n}\theta_ix_i-y)$

$=(h_{\theta}(x)-y)x_j$

更新$\theta$：$\theta_j:=\theta_j+\alpha\cdot(y^{(i)}-h_{\theta}(x^{(i)}))x_j^{(i)}$



+ <font color=red>分类：</font>
  + 随机梯度下降 SGD 

  <font color=red>一条样本一条样本的更新</font>$\theta$值（上面更新$\theta$的式子就是SGD）

  ​	**优点：**训练速度快；

  ​	**缺点：**准确度下降，并不是全局最优；不易于并行实现。
  
  + 批量梯度下降 BGD

  <font color=red>所有样本一起更新</font>$\theta$值

  $\theta_j=\theta_j+\alpha \cdot \sum_{i=0}^{m}(y^{(i)}-h_{\theta}(x^{(i)}))\cdot x_j^{(i)}$

  ​	**优点：**全局最优解；易于并行实现；

  ​	**缺点：**当样本数目很多时，训练过程会很慢。

  + mini batch 梯度下降

  <font color=red>取样本的一部分来更新</font>$\theta$值







