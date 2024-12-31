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

## 李杜 | 苏辛

#### 训练

###### 数据集

* 原始文件嵌入以后生成嵌入文件作为训练数据

```
# 原始文件
train/诗词1.json
train/诗词2.json
val/诗词1.json
val/诗词2.json

# 嵌入文件
train/embedding.model
val/embedding.model
```

#### 推理

## 吴道子（图片风格迁移）

#### 训练

###### 数据集

```
# 原始文件
train/图片1.source.jpg
train/图片2.source.jpg
val/图片1.source.jpg
val/图片2.source.jpg

# 目标文件
train/图片1.target.jpg
train/图片2.target.jpg
val/图片1.target.jpg
val/图片2.target.jpg
```

#### 推理

## 顾恺之（图片内容生成）

#### 训练

###### 数据集

```
# 描述文件
train/图片1.json
train/图片2.json
val/图片1.json
val/图片2.json

# 图片文件
train/图片1.jpg
train/图片2.jpg
val/图片1.jpg
val/图片2.jpg
```

#### 推理

## 师旷（音频风格迁移）

#### 训练

###### 数据集

原始音频文件要解析成`PCM`文件作为训练数据，可以直接提供单声道采样率`48000`的`s16le`格式的`PCM`文件，音频需要手动对齐保证训练效果。

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
```

#### 推理

## 李龟年（音频内容生成）

#### 训练

###### 数据集

原始文件解析成为`PCM`文件作为训练数据

```
# 描述文件
train/音频1.json
train/音频2.json
val/音频1.json
val/音频2.json

# 原始文件
train/音频1.mp3
train/音频2.mp3
val/音频1.mp3
val/音频2.mp3

# PCM文件
train/音频1.pcm
train/音频2.pcm
val/音频1.pcm
val/音频2.pcm
```

#### 推理

## 关汉卿（视频风格迁移）

#### 训练

###### 数据集

```
# 原始视频
train/视频1.source.mp4
train/视频2.source.mp4
val/视频1.source.mp4
val/视频2.source.mp4

# 目标视频
train/视频1.target.mp4
train/视频2.target.mp4
val/视频1.target.mp4
val/视频2.target.mp4
```

#### 推理

## 汤显祖（视频动作预测）

#### 训练

###### 数据集

```
train/视频1.mp4
train/视频2.mp4
val/视频1.mp4
val/视频2.mp4
```
