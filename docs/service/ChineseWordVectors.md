# Chinese-Word-Vectors

`Embedding`相关功能

## 部署

```
# 目录
mkdir -p /data ; cd $_

# 环境
conda create --name Chinese-Word-Vectors python=3.11
conda activate Chinese-Word-Vectors

# 下载
git clone https://github.com/Embedding/Chinese-Word-Vectors.git

# 依赖
pip install numpy
pip install gensim

# 配置
# https://github.com/Embedding/Chinese-Word-Vectors/blob/master/README_zh.md
# 下载相应向量数据

# 退出
conda deactivate
```

## 常用功能

```
from gensim.models.keyedvectors import KeyedVectors
w2v_model = KeyedVectors.load_word2vec_format("sgns.sikuquanshu.bigram.bz2", binary=False,unicode_errors='ignore')
print(w2v_model)
ret = w2v_model.get_vector('李')
print(ret)
sim = w2v_model.most_similar('夫')
print(sim)

```

## 相关链接

* https://github.com/Embedding/Chinese-Word-Vectors

## 注意事项
