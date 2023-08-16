# 人工智能

* 人工智能（AI）
* 机器学习（ML）
* 神经网络（NN）
* 深度学习（DL）

* 传统任务：回归、分类、聚类（高维特征）
* 深度学习：计算机视觉、自然语言处理

* 监督学习：分类、回归
* 非监督学习：聚类
* 强化学习

## 回归

### 线性回归

线性回归（Linear Regression）可能是最流行的机器学习算法。线性回归就是要找一条直线，并且让这条直线尽可能地拟合散点图中的数据点。

[线性回归](https://baike.baidu.com/item/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92)

### 多项式回归

## 分类

### Logistic回归（逻辑回归算法）

逻辑回归（Logistic regression）与线性回归类似，通常用来解决二分类问题，也可以使用softmax方法处理多分类问题。

[逻辑回归](https://baike.baidu.com/item/logistic%E5%9B%9E%E5%BD%92)

## 聚类

### K-Means（K均值聚类算法）

K均值聚类算法是一种迭代求解的聚类分析算法

[K均值聚类算法](https://baike.baidu.com/item/K%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95/15779627)

### DBSCAN（基于密度的聚类算法）

聚类算法

## 神经网络

## 深度学习

### 岭回归

### 决策树

* 回归树（Classification And Regression Tree CART）
* ID3 (Iterative Dichotomiser 3)
* C4.5
* Chi-squared Automatic Interaction Detection(CHAID)
* Decision Stump
* 随机森林（Random Forest）
* 多元自适应回归样条（MARS）
* 梯度推进机（Gradient Boosting Machine GBM）

决策树（Decision Trees）可用于回归和分类任务。

[决策树](https://baike.baidu.com/item/%E5%86%B3%E7%AD%96%E6%A0%91/10377049)

### 贝叶斯方法

* 朴素贝叶斯算法
* 平均单依赖估计（Averaged One-Dependence Estimators AODE）
* Bayesian Belief Network（BBN）

贝叶斯方法算法是基于贝叶斯定理的一类算法，主要用来解决分类和回归问题。

### 朴素贝叶斯

朴素贝叶斯（Naive Bayes）是基于贝叶斯定理与特征条件独立假设的分类方法。

[朴素贝叶斯](https://baike.baidu.com/item/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF/4925905)

### 基于核的算法

* 支持向量机（Support Vector Machine SVM）
* 径向基函数（Radial Basis Function RBF）
* 线性判别分析（Linear Discriminate Analysis LDA）

基于核的算法把输入数据映射到一个高阶的向量空间， 在这些高阶向量空间里， 有些分类或者回归问题能够更容易的解决。

### 支持向量机

支持向量机（SVM）是一种用于分类问题的监督算法。

[支持向量机](https://baike.baidu.com/item/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA)

### KNN算法（邻近算法）

K最近邻分类算法是数据挖掘分类技术中最简单的方法之一

[邻近算法](https://baike.baidu.com/item/%E9%82%BB%E8%BF%91%E7%AE%97%E6%B3%95/1151153)


### 随机森林

随机森林（Random Forest）是一种非常流行的集成机器学习算法。

[随机森林](https://baike.baidu.com/item/%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97/1974765)

### 降低维度算法

* 主成份分析（Principle Component Analysis PCA）
* 偏最小二乘回归（Partial Least Square Regression PLS）
* Sammon映射，多维尺度（Multi-Dimensional Scaling MDS）
* 投影追踪（Projection Pursuit）

像聚类算法一样，降低维度算法试图分析数据的内在结构，不过降低维度算法是以非监督学习的方式试图利用较少的信息来归纳或者解释数据。这类算法可以用于高维数据的可视化或者用来简化数据以便监督式学习使用。

### 人工神经网络

* 感知器神经网络（Perceptron Neural Network）
* 反向传递（Back Propagation）
* Hopfield网络
* 自组织映射（Self-Organizing Map SOM）
* 学习矢量量化（Learning Vector Quantization LVQ）

人工神经网络算法模拟生物神经网络，是一类模式匹配算法。通常用于解决分类和回归问题。

### 深度学习

* 受限波尔兹曼机（Restricted Boltzmann Machine RBN）
* Deep Belief Networks（DBN）
* 卷积网络（Convolutional Network）
* 堆栈式自动编码器（Stacked Auto-encoders）

深度学习算法是对人工神经网络的发展。

### 集成算法

* Boosting
* Bootstrapped Aggregation（Bagging）
* AdaBoost
* 堆叠泛化（Stacked Generalization Blending）
* 梯度推进机（Gradient Boosting Machine GBM）
* 随机森林（Random Forest）

集成算法用一些相对较弱的学习模型独立地就同样的样本进行训练，然后把结果整合起来进行整体预测。集成算法的主要难点在于究竟集成哪些独立的较弱的学习模型以及如何把学习结果整合起来。

### 梯度下降

随机梯度下降、批量梯度下降

### 正则化方法

* Elastic Net
* Ridge Regression
* Least Absolute Shrinkage and Selection Operator（LASSO）

正则化方法是其他算法（通常是回归算法）的延伸，根据算法的复杂度对算法进行调整。正则化方法通常对简单模型予以奖励而对复杂算法予以惩罚。

## 学习资料

* https://zhuanlan.zhihu.com/p/402192877
* https://yuanzhuo.bnu.edu.cn/article/820
* https://www.bilibili.com/read/cv13813224/
