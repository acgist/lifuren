# 计划

> *√=完成、○-进行中、#-未开始、?-待定、--忽略*

## 主要功能

|任务|当前状态|详细功能|
|:--|:--:|:--|
|诗词|○|李杜、苏辛|
|图片|○|吴道子、顾恺之|
|音频|○|师旷、李龟年|
|视频|○|关汉卿、汤显祖|

## 详细功能

|任务|当前状态|详细描述|
|:--|:--:|:--|
|李杜|#|作诗|
|苏辛|○|填词|
|吴道子|#|图片风格迁移|
|顾恺之|?|图片内容生成|
|师旷|○|音频风格迁移|
|李龟年|?|音频内容生成|
|关汉卿|#|视频风格迁移|
|汤显祖|○|视频动作预测|
|诗词嵌入|○|-|
|模型部署|○|`ONNX`/`TensorRT`|
|`CLI`接口|○|命令行接口|
|`GUI`接口|○|图形界面接口|
|`REST`接口|○|只有推理接口|
|模型微调|#|-|
|模型量化|#|-|

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

* `source`为原始图片`target`为目标文件

```
train/图片1.source.jpg
train/图片1.target.jpg
train/图片2.source.jpg
train/图片2.target.jpg
val/图片1.source.jpg
val/图片1.target.jpg
val/图片2.source.jpg
val/图片2.target.jpg
```

#### 推理

## 顾恺之（图片内容生成）

#### 训练

###### 数据集

* `json`为相同文件名的图片描述

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

* `source`为原始图片`target`为目标文件
* 原始文件解析成为`PCM`文件作为训练数据

```
# 原始文件
train/音频1.source.mp3
train/音频1.target.mp3
train/音频2.source.mp3
train/音频2.target.mp3
val/音频1.source.mp3
val/音频1.target.mp3
val/音频2.source.mp3
val/音频2.target.mp3

# PCM文件
train/音频1.source.pcm
train/音频1.target.pcm
train/音频2.source.pcm
train/音频2.target.pcm
val/音频1.source.pcm
val/音频1.target.pcm
val/音频2.source.pcm
val/音频2.target.pcm
```

#### 推理

## 李龟年（音频内容生成）

#### 训练

###### 数据集

* `json`为相同文件名的音频描述
* 原始文件解析成为`PCM`文件作为训练数据

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

* `source`为原始图片`target`为目标文件

```
train/视频1.source.mp4
train/视频1.target.mp4
train/视频2.source.mp4
train/视频2.target.mp4
val/视频1.source.mp4
val/视频1.target.mp4
val/视频2.source.mp4
val/视频2.target.mp4
```

#### 推理（视频动作预测）

## 汤显祖

#### 训练

###### 数据集

```
train/视频1.mp4
train/视频2.mp4
val/视频1.mp4
val/视频2.mp4
```

#### 推理

## 代码规范

* 工具类不要用前缀后缀
* 宏定义开头必须使用`LFR_`
* 尽量使用智能指针和避免使用裸指针
* 除了变量之外避免使用复数（方法、类名）
* 在可能出现溢出的地方使用`LL`/`ULL`后缀
* 头文件宏定义`LFR_HEADER_MODULE_PATH_FILENAME_HPP`
* 全局方法必须使用命名空间`lifuren::module | lifuren::filename`
