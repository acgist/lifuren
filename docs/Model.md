# 模型

|模型|当前状态|详细描述|
|:--|:--:|:--|
|巴赫|○|音频识谱|
|肖邦|○|简谱识谱|
|师旷|○|音频风格迁移|
|莫扎特|○|五线谱识谱|
|贝多芬|○|乐谱钢琴指法|

## 数据集

```
# 训练数据集
train/...
# 验证数据集
val/...
# 测试数据集
test/...
```

## 识谱方案

#### 方案一

直接使用`YOLO`将音符和音高直接识别，缺点需要标记所有音高，数据集制作耗时。

#### 方案二

先使用`YOLO`将音符识别出来，然后配合乐谱训练音高识别模型，缺点极度依赖`YOLO`识别度，不然匹配音符数量匹配不上。

## 训练

训练使用`CLI`

## 预测

预测使用`CLI`/`GUI`

## 嵌入文件

一些数据集预处理非常耗时，提前生成嵌入文件可以减小训练时加载数据的时间。

## 师旷

```
音频特征：音高 + 音调 + 音色
实现原理：原始 + 特征 = 目标
```

#### 数据集

原始文件最好使用纯人声音频，可以使用`UVR5`提取纯净人声。

```
train/音频1.mp3
train/音频2.mp3
val/音频1.mp3
val/音频2.mp3
```

## 巴赫

暂时支持人声

#### 数据集

```
train/音频1.mp3
train/乐谱1.xml
train/音频2.mp3
train/乐谱2.xml
val/音频1.mp3
val/乐谱1.xml
val/音频2.mp3
val/乐谱2.xml
```

## 肖邦

#### 数据集

```
train/简谱1.jpg
train/乐谱1.xml
train/简谱2.jpg
train/乐谱2.xml
val/简谱1.jpg
val/乐谱1.xml
val/简谱2.jpg
val/乐谱2.xml
```

## 莫扎特

#### 数据集

```
train/五线谱1.jpg
train/乐谱1.xml
train/五线谱2.jpg
train/乐谱2.xml
val/五线谱1.jpg
val/乐谱1.xml
val/五线谱2.jpg
val/乐谱2.xml
```

## 贝多芬

#### 数据集

* https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/

```
train/乐谱1.xml
train/指法1.xml
train/乐谱2.xml
train/指法2.xml
val/乐谱1.xml
val/指法1.xml
val/乐谱2.xml
val/指法2.xml
```
