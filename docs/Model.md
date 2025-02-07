# 模型

## 数据集

```
# 训练数据集
train/...
# 验证数据集
val/...
# 测试数据集
test/...
```

## 嵌入文件

一些数据集预处理非常耗时，提前生成嵌入文件可以减小训练时加载数据的时间。

## 师旷（音频风格迁移）

#### 数据集

原始音频文件要解析成`PCM`文件作为训练数据，可以直接提供单声道`48000`采样率`s16le`格式的`PCM`文件，音频需要手动对齐保证训练效果。

```
# 原始文件
train/音频1.source.mp3
train/音频2.source.mp3
val/音频1.source.mp3
val/音频2.source.mp3

# 目标文件
train/音频1.target.mp3
train/音频2.target.mp3
val/音频1.target.mp3
val/音频2.target.mp3

# PCM原始文件
train/音频1.source.pcm
train/音频2.source.pcm
val/音频1.source.pcm
val/音频2.source.pcm

# PCM目标文件
train/音频1.target.pcm
train/音频2.target.pcm
val/音频1.target.pcm
val/音频2.target.pcm

# 嵌入文件
train/embedding.model
val/embedding.model
```

#### 结构

#### 训练

#### 推理

## 吴道子（视频动作预测）

#### 数据集

```
train/视频1.mp4
train/视频2.mp4
val/视频1.mp4
val/视频2.mp4
```

#### 结构

#### 训练

#### 推理
