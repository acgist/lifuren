# 人工智能

特征 + 模型

## 流程

```
原始数据 -> 特征工程 -> 设计模型 -> 训练模型（训练、评估、测试） -> 部署模型
```

## 境界

#### 初窥门径

了解基本知识，能够使用现有模型进行简单的参数调优。

#### 登堂入室

了解各种基本算法和网络结构，能够设计训练简单模型。

#### 炉火纯青

熟悉各种数据特征工程，能够针对不同任务独立设计合适的模型。

#### 登峰造极

掌握相关算法的数学原理（微积分、概率论、信息论、统计学、线性代数等等），能够独立设计实现相关算法。

## 分类

* 人工智能（AI）
* 机器学习（ML）
* 神经网络（NN）
* 深度学习（DL）

#### 学习方式

* 有监督学习
* 无监督学习
* 半监督学习
* 自监督学习

* 元学习
* 迁移学习
* 强化学习

#### 算法模式

* 判别式模型
* 生成式模型

## 应用

* 传统任务：回归、分类、聚类
* 深度学习：机器视觉、自然语言

#### 机器视觉（CV）

图像分类、目标分割、目标检测、目标识别、目标跟踪等等

#### 自然语言（NLP）

文本分类、情感分析、机器翻译、自动摘要、问答系统、文本生成等等

#### 音频相关

语音识别、情感识别、语音克隆、音频降噪、歌声转换、文本转语音等等

## 回归算法（regression）

#### 线性回归算法（Linear Regression）
#### 多项式回归算法（Polynomial Regression）

## 分类算法（classification）

#### 邻近算法（KNN）
#### 支持向量机（SVM）
#### 随机森林算法（Random Forest）
#### 逻辑回归算法（Logistic Regression）
#### 隐马尔可夫模型（HMM）
#### Softmax逻辑回归算法（Softmax Regression）

## 聚类算法

#### K均值聚类算法（K-Means）
#### 基于密度的聚类算法（DBSCAN）

## 神经网络算法 & 深度学习算法 & 生成网络算法

#### 人工神经网络（ANN）
#### 深度神经网络（DNN）
#### 卷积神经网络（CNN）
###### 视觉几何组（VGG）
###### 残差神经网络（ResNet）
###### 稠密连接⽹络（DenseNET）
#### UNet
#### YOLO
#### 循环神经网络（RNN）
###### 门控循环单元（GRU）
###### 长短时记忆网络（LSTM）
#### 图神经网络（GNN）
#### 生成对抗网络（GAN）
###### 循环生成对抗网络（CycleGAN）
###### 基于风格对抗网络（StyleGAN）
#### 扩散模型（Diffusion）
###### VAE
###### CLIP
###### Flux
###### LoRA
#### 注意力机制
#### 大语言模型（LLM）
###### gemma
###### llama
###### mamba
#### 多模态大模型（MLLM）
#### 视觉语言模型（VLM）
#### 视觉语言动作多模态模型（VLA）

## 术语

#### 早停（early stopping）
#### 滤波器（filter）
#### 感受野（receptive field）
#### 丢弃法（Dropout）
#### 归一化（Normalization）
#### 正则化（Regularization）
#### 标准化（Standardization）
#### 学习率（learning rate）
#### 超参数（hyperparameter）
#### 过拟合（overfitting）
#### 欠拟合（underfitting）
#### 计算图
#### 预训练
#### 网络压缩
#### 模型量化（Quantization）
#### 模型微调（Fine-tune）
#### 模型精调
#### 激活函数
###### ReLU
###### Tanh
###### Sigmod
#### 优化算法
###### Adam
###### AdamW
###### Adamax
###### AdaGrad
###### AdaDelta
###### 批量梯度下降法（BGD）
###### 随机梯度下降法（SGD）
#### 损失函数
###### 均方误差（MSE）
###### 交叉熵损失（Cross Entropy Loss）
###### 均方根误差（RMSE）
###### 平均绝对值误差（MAE）
###### 负对数似然损失（NLL）
#### 训练误差
#### 泛化误差
#### 特征工程
#### 共享参数
#### 数据增强
#### 模型评价
#### 正向传播
#### 反向传播
#### 权重衰减
#### 梯度消失
#### 梯度爆炸
#### 计划采样（Scheduled Sampling）
#### 检索增强生成（RAG）
#### 自然语言推理（NLI）

## 训练框架

* [PyTorch](https://github.com/pytorch/pytorch)
* [MindSpore](https://github.com/mindspore-ai/mindspore)
* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [PaddlePaddle](https://github.com/PaddlePaddle/PaddleHub)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)

## 部署框架

* [GGML](https://github.com/ggerganov/ggml)
* [OpenVINO](https://github.com/openvinotoolkit/openvino)
* [TensorRT](https://developer.nvidia.com/tensorrt)
* [MediaPipe](https://github.com/google-ai-edge/mediapipe)
* [OnnxRuntime](https://github.com/microsoft/onnxruntime)

## 模型结构

[Netron](https://netron.app)

## 模型下载

* https://ollama.com
* https://huggingface.co

## 数据集下载

* https://image-net.org
* https://www.openslr.org

## 学习资料

* https://zh.d2l.ai
* https://github.com/datawhalechina
* https://github.com/datawhalechina/fun-rec
* https://github.com/datawhalechina/easy-rl
* https://github.com/datawhalechina/self-llm
* https://github.com/datawhalechina/pumpkin-book
* https://github.com/datawhalechina/llm-cookbook
* https://github.com/datawhalechina/llm-universe
* https://github.com/datawhalechina/leedl-tutorial
