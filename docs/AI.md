# 人工智能

特征 + 模型

## 流程

原始数据 -> 特征工程 -> 建模（训练、评估） -> 模型

## 分类

* 人工智能（AI）
* 机器学习（ML）
* 神经网络（NN）
* 深度学习（DL）

#### 学习方式

* 强化学习
* 有监督学习
* 无监督学习
* 半监督学习

## 应用

* 传统任务：回归、分类、聚类
* 深度学习：机器视觉、自然语言

#### 机器视觉

图像分类、图像分割、目标检测等等

#### 自然语言

语音识别、机器翻译、自动摘要、观点提取、文本分类、问题回答等等

## 回归算法（regression）

#### 线性回归算法（Linear Regression）
#### 多项式回归算法（Polynomial Regression）

## 分类算法（classification）

#### 邻近算法（KNN）
#### 支持向量机（SVM）
#### 隐马尔可夫模型（HMM）
#### 随机森林算法（Random Forest）
#### 逻辑回归算法（Logistic Regression）
#### Softmax逻辑回归算法（Softmax Regression）

## 聚类算法

#### K均值聚类算法（K-Means）
#### 基于密度的聚类算法（DBSCAN）

## 神经网络算法 & 深度学习算法 & 生成网络算法

#### 人工神经网络（ANN）

#### 卷积神经网络（CNN）

机器视觉

* 核
* 步幅
* 填充

###### 视觉几何组（VGG）
###### 残差神经网络（ResNet）
###### 稠密连接⽹络（DenseNET）

图像分类

#### 循环神经网络（RNN）

自然语言

###### 门控循环单元（GRU）
###### 长短时记忆网络（LSTM）

#### UNet

图像分割

#### YOLO

目标检测

#### 生成对抗网络（GAN）

* https://github.com/podgorskiy/ALAE
* https://github.com/junyanz/CycleGAN
* https://github.com/TencentARC/GFPGAN
* https://github.com/eladrich/pixel2style2pixel
* https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

###### 循环生成对抗网络（CycleGAN）
###### 基于风格对抗网络（StyleGAN）

#### VQGAN

* https://github.com/CompVis/taming-transformers

#### mamba

对比RNN/Transformer结构

* https://github.com/state-spaces/mamba

#### 大语言模型（LLM）

* https://github.com/THUDM/ChatGLM3
* https://github.com/THUDM/ChatGLM-6B
* https://github.com/THUDM/ChatGLM2-6B
* https://github.com/Stability-AI/StableLM
* https://github.com/tatsu-lab/stanford_alpaca

#### whisper

* https://github.com/openai/whisper
* https://github.com/ggerganov/whisper.cpp

#### Transformer

* https://github.com/huggingface/transformers

###### T5
###### GPT
###### BERF
###### Self-Attention

#### gemma

* https://github.com/google/gemma.cpp
* https://github.com/google/gemma_pytorch

#### llama

* https://github.com/ollama/ollama
* https://github.com/meta-llama/llama
* https://github.com/meta-llama/llama3
* https://github.com/ggerganov/llama.cpp
* https://github.com/LlamaFamily/Llama-Chinese
* https://github.com/ymcui/Chinese-LLaMA-Alpaca
* https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
* https://github.com/ymcui/Chinese-LLaMA-Alpaca-3

#### 扩散模型（Diffusion）

* https://github.com/huggingface/diffusers
* https://github.com/CompVis/latent-diffusion
* https://github.com/CompVis/stable-diffusion
* https://github.com/leejet/stable-diffusion.cpp
* https://github.com/Stability-AI/stablediffusion

#### CLIP

* https://github.com/openai/clip
* https://github.com/monatis/clip.cpp

#### Embedding

###### GloVe
###### Word2Vec
###### FastText

* https://github.com/supabase/supabase
* https://github.com/chroma-core/chroma
* https://github.com/Embedding/Chinese-Word-Vectors

#### 其他

* https://github.com/lencx/ChatGPT
* https://github.com/geekan/MetaGPT
* https://github.com/ggerganov/ggml
* https://github.com/xtekky/gpt4free
* https://github.com/salesforce/BLIP
* https://github.com/reworkd/AgentGPT
* https://github.com/opendatalab/MinerU
* https://github.com/TencentARC/PhotoMaker
* https://github.com/Akegarasu/lora-scripts
* https://github.com/langchain-ai/langchain
* https://github.com/KichangKim/DeepDanbooru
* https://github.com/Significant-Gravitas/AutoGPT
* https://github.com/Stability-AI/generative-models
* https://github.com/PlexPt/awesome-chatgpt-prompts-zh
* https://github.com/chatchat-space/Langchain-Chatchat
* https://github.com/AUTOMATIC1111/stable-diffusion-webui

## 迁移学习
## 模型量化

## 模型微调（Fine-tune）

* https://github.com/huggingface/peft

## 模型精调
## 激活函数

激活函数的作用是在神经网络中引入非线性性质，使其能够学习复杂的非线性关系。
常用的激活函数包括`Sigmoid`、`ReLU`、`Tanh`等等。

## 优化算法

#### 批量梯度下降法（BGD）
#### 随机梯度下降法（SGD）
#### Adam
#### AdamW

ADMA + L2

#### Adamax
#### AdaGrad
#### AdaDelta

## 损失函数

* 经验风险损失函数
* 结构风险损失函数

#### 均方误差（MSE）

计算预测值与实际值之间的平方差，并求取平均值。常用于回归问题。

#### 均方根误差（RMSE）

均方根误差是均方误差的平方根。它具有与均方误差相同的特性，但对异常值更加敏感。

#### 平均绝对值误差（MAE）

计算预测值与实际值之间的绝对差，并求取平均值。常用于回归问题，对异常值不敏感。

#### 交叉熵损失（Cross Entropy Loss）

衡量预测概率分布与实际标签之间的差异。常用于分类问题。
常用的交叉熵损失有二分类交叉熵（Binary Cross Entropy）和多分类交叉熵（Categorical Cross Entropy）。

#### 负对数似然损失（NLL）

适用于概率预测问题，即给定输入条件下的概率预测。
常用于文本生成、语言模型等等任务。

## 术语

#### 早停（early stopping）

#### 正则化（Regularization）

###### L1
###### L2

#### 丢弃法（Dropout）
#### 批量归一化（Batch Normalization）

#### 特征归一化（feature normalization）

* Z值归一化（Z-score normalization）也叫标准化（standardization）

#### 感受野（receptive field）
#### 滤波器（filter）
#### 学习率（learning rate）
#### 超参数（hyperparameter）

#### 过拟合

* 增加训练集（数据增强）
* 限制模型（减小参数、减小特征、早停、正则化、丢弃法）

#### 欠拟合

#### 零界点

梯度为零的点（鞍点、局部极小值）

#### 计算图

* 动态图
* 静态图

#### 特征工程

###### 数值
###### 文本
###### 分类变量
###### 特征缩放
###### 数据降维
###### 数据分箱
###### 图像裁剪
###### 图像增⼴

#### 共享参数（parameter sharing）
#### 数据增强
#### 模型评价
#### 正向传播
#### 反向传播
#### 权重衰减

常用的正则化技术，减少模型的过拟合现象。

#### 梯度消失

#### 分类函数

* 二分类：sigmoid
* 多分类：softmax

#### 梯度爆炸
#### prompt
#### Tokenizer
#### Embedding
#### Normalization（归一化）

1. (x - min(x))  / (max(x) - min(x)) [ 0, 1]
2. (x - mean(x)) / (max(x) - min(x)) [-1, 1]

#### Standardization（标准化）

1. (x - mean(x)) / std(x)

## 训练框架

#### PyTorch
#### TensorFlow
#### PaddlePaddle

## 模型工具

#### ONNX
#### Netron
#### TensorRT

## 模型下载

* https://huggingface.co
* https://huggingface.co/stabilityai
* https://huggingface.co/stabilityai/stable-diffusion-2-1-base

## 学习资料

* https://zh-v2.d2l.ai/
* https://github.com/datawhalechina
* https://github.com/datawhalechina/fun-rec
* https://github.com/datawhalechina/easy-rl
* https://github.com/datawhalechina/self-llm
* https://github.com/datawhalechina/pumpkin-book
* https://github.com/datawhalechina/llm-cookbook
* https://github.com/datawhalechina/llm-universe
* https://github.com/datawhalechina/leedl-tutorial
