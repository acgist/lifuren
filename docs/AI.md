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

## 编程境界

#### 初窥门径
能够使用一门语言完成一些任务，排查一些常见错误。
#### 登堂入室
掌握一门编程语言，熟悉网络编程、多线程编程以及常见框架和设计模式。
#### 炉火纯青
掌握各种框架原理，能够自己实现框架，熟悉各种代码分析优化方法，编写代码高效执行。
#### 登峰造极
深入理解计算机原理，不拘泥于编程语言，了解多种业务领域，对各种算法模式架构随心所欲信手拈来。

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

#### 算法模式

* 判别式模型
* 生成式模型

## 应用

* 传统任务：回归、分类、聚类
* 深度学习：机器视觉、自然语言

#### 机器视觉（CV）

图像分类、目标分割、目标检测、目标识别、目标跟踪等等

#### 自然语言（NLP）

文本分类、情感分析、机器翻译、观点提取、自动摘要、问答系统、文本生成等等

#### 音频相关

语音识别（语音转文本）、情感识别、语音克隆（拟声）、音频降噪、歌声转换、文本转语音等等

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

#### 深度神经网络（DNN）

#### 卷积神经网络（CNN）

机器视觉

###### 视觉几何组（VGG）
###### 残差神经网络（ResNet）
###### 稠密连接⽹络（DenseNET）

#### 循环神经网络（RNN）

自然语言

###### 门控循环单元（GRU）
###### 长短时记忆网络（LSTM）

#### 图神经网络（GNN）

非欧几里得数据

#### UNet

图像分割

#### YOLO

目标检测

* https://github.com/ultralytics/yolov5
* https://github.com/ultralytics/ultralytics

#### 生成对抗网络（GAN）

* https://github.com/podgorskiy/ALAE
* https://github.com/junyanz/CycleGAN
* https://github.com/TencentARC/GFPGAN
* https://github.com/eladrich/pixel2style2pixel
* https://github.com/TachibanaYoshino/AnimeGANv2
* https://github.com/TachibanaYoshino/AnimeGANv3
* https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

###### 循环生成对抗网络（CycleGAN）
###### 基于风格对抗网络（StyleGAN）

#### VQGAN

* https://github.com/CompVis/taming-transformers

#### 大语言模型（LLM）

* https://github.com/THUDM/ChatGLM3
* https://github.com/THUDM/ChatGLM-6B
* https://github.com/THUDM/ChatGLM2-6B
* https://github.com/Stability-AI/StableLM
* https://github.com/tatsu-lab/stanford_alpaca

#### 视觉语言模型（VLM）

#### 视觉语言动作多模态模型（VLA）

#### whisper

语音识别

* https://github.com/openai/whisper
* https://github.com/ggerganov/whisper.cpp

#### Transformer

自然语言

* https://github.com/huggingface/transformers

###### T5
###### GPT
###### BERT
###### Self-Attention

#### GLUE

通用语言理解评估（GLUE）基准

#### mamba

类似Transformer架构（竞争关系）

* https://github.com/state-spaces/mamba

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

#### VAE

#### LoRA

#### CLIP

* https://github.com/openai/clip
* https://github.com/monatis/clip.cpp

#### Flux

* https://github.com/black-forest-labs/flux

#### Embedding

* Token Embeddings
* Segment Embeddings
* Position Embeddings

###### GloVe
###### Word2Vec
###### FastText

* https://github.com/supabase/supabase
* https://github.com/chroma-core/chroma
* https://github.com/Embedding/Chinese-Word-Vectors

#### 语音识别（ASR）

#### 实时变声（RVC）

* https://github.com/babysor/MockingBird
* https://github.com/CorentinJ/Real-Time-Voice-Cloning

#### 歌声转换（SVC）

* https://github.com/prophesier/diff-SVC
* https://github.com/svc-develop-team/so-vits-svc

#### 文本转语音（TTS）

###### 歌声合成（SVS）

#### 其他

* https://github.com/lencx/ChatGPT
* https://github.com/geekan/MetaGPT
* https://github.com/xtekky/gpt4free
* https://github.com/salesforce/BLIP
* https://github.com/reworkd/AgentGPT
* https://github.com/opendatalab/MinerU
* https://github.com/TencentARC/PhotoMaker
* https://github.com/Akegarasu/lora-scripts
* https://github.com/langchain-ai/langchain
* https://github.com/comfyanonymous/ComfyUI
* https://github.com/KichangKim/DeepDanbooru
* https://github.com/Significant-Gravitas/AutoGPT
* https://github.com/Stability-AI/generative-models
* https://github.com/PlexPt/awesome-chatgpt-prompts-zh
* https://github.com/chatchat-space/Langchain-Chatchat
* https://github.com/AUTOMATIC1111/stable-diffusion-webui

## 迁移学习

#### 领域偏移

## 强化学习
## 元学习
## 终身学习

## 网络压缩

* 网络剪枝
* 知识蒸馏
* 参数量化
* 动态计算
* 网络架构设计

## 模型量化（Quantization）

## 模型微调（Fine-tune）

* https://github.com/huggingface/peft

## 模型精调
## 激活函数

激活函数的作用是在神经网络中引入非线性性质，使其能够学习复杂的非线性关系。
常用的激活函数包括`Sigmoid`、`ReLU`、`Tanh`等等。

* 回归：ReLU
* 分类：Tanh

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

###### L1范数
###### L2范数
###### LP范数

#### 丢弃法（Dropout）
#### 归一化（Normalization）

* Z值归一化（Z-score normalization）也叫标准化（standardization）

###### 批量归一化（Batch Normalization）
###### 特征归一化（Feature Normalization）

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

#### 预训练

#### 特征工程

* 数值
* 文本
* 归一化
* 离散化
* 特征清洗：异常处理、数据采样
* 分类变量
* 特征缩放
* 数据变换：log、指数、Box-Cox
* 数据降维：PCA、LDA
* 特征交互
* 数据分箱
* 图像裁剪
* 图像增⼴

#### 共享参数（parameter sharing）
#### 数据增强
#### 模型评价
#### 正向传播
#### 反向传播
#### 权重衰减

常用的正则化技术，减少模型的过拟合现象。

#### 梯度消失

#### 梯度爆炸

#### 分类函数

* 二分类：sigmoid
* 多分类：softmax

#### 计划采样（Scheduled Sampling）

不要总是训练对的数据，给一些错误的噪声。

#### 检索增强生成（RAG）
#### 自然语言推理（NLI）
#### prompt
#### Tokenizer
#### Embedding

#### Normalization（归一化）

提升模型精度：归一化后，不同维度之间的特征在数值上有一定比较性，可以大大提高分类器的准确性。

```
1. (x - min(x))  / (max(x) - min(x)) [ 0, 1]
2. (x - mean(x)) / (max(x) - min(x)) [-1, 1]
```

#### Standardization（标准化）

加速模型收敛：标准化后，最优解的寻优过程明显会变得平缓，更容易正确的收敛到最优解。如下图所示：

```
1. (x - mean(x)) / std(x)
```

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
* [OnnxRuntime](https://github.com/microsoft/onnxruntime)

## 模型结构

[Netron](https://netron.app)

## 模型下载

* https://huggingface.co
* https://huggingface.co/stabilityai
* https://huggingface.co/stabilityai/stable-diffusion-2-1-base

## 数据集下载

* https://www.openslr.org

## 学习资料

* https://zh-v2.d2l.ai
* https://github.com/datawhalechina
* https://github.com/datawhalechina/fun-rec
* https://github.com/datawhalechina/easy-rl
* https://github.com/datawhalechina/self-llm
* https://github.com/datawhalechina/pumpkin-book
* https://github.com/datawhalechina/llm-cookbook
* https://github.com/datawhalechina/llm-universe
* https://github.com/datawhalechina/leedl-tutorial
