# Conda

`Anaconda`是一个用于科学计算的`Python`发行版，包含了众多流行的科学计算、数据分析的`Python`包。

## 项目地址

* https://anaconda.org/anaconda
* https://github.com/conda/conda
* https://www.anaconda.com/download/success
* https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
* https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

## 项目部署

```
# 下载安装包
# https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

# 修改配置
vim ~/.condarc

---
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
---

# 清除索引缓存
conda clean -i
```

## 常用功能

```
# 更新Conda
conda update conda
conda update anaconda
conda update --all

# 更新模块
conda update  xxxx
# 安装模块
conda install xxxx

## 创建虚拟环境
conda create --name 环境名称 python=3.11
## 激活某个环境
conda activate 环境名称
## 退出当前环境
conda deactivate
## 删除某个环境
conda remove --name 环境名称 --all
## 复制某个环境
conda create --name 环境名称 --clone 环境名称
## 列出所有的环境
conda env list
```

## 相关项目

* https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/

## 注意事项
