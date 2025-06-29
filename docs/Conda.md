# Conda

`Python`环境管理

## 部署

```
# 目录
mkdir -p /data/conda ; cd $_

# 下载
# wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

# 安装
sh Anaconda3-2024.06-1-Linux-x86_64.sh

# 升级
sh Anaconda3-2024.06-1-Linux-x86_64.sh -u

# 目录
# /data/conda/anaconda3

# 配置
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
  pytorch    : https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
---

# 环境变量
. ~/.bashrc

# 验证
conda -V
```

## 常用功能

```
# 更新软件
conda update conda
conda update anaconda
conda update --all

# 安装模块
conda install xxxx
# 更新模块
conda update  xxxx

# 创建虚拟环境
conda create --name 环境名称 python=3.12
# 复制某个环境
conda create --name 环境名称 --clone 环境名称
# 激活某个环境
conda activate 环境名称
# 退出当前环境
conda deactivate
# 删除某个环境
conda remove --name 环境名称 --all
# 列出所有环境
conda env list
# 清除索引缓存
conda clean -i
```

## 相关链接

* https://anaconda.org/anaconda
* https://github.com/conda/conda
* https://www.anaconda.com/download/success
* https://mirrors.tuna.tsinghua.edu.cn/help/anaconda
* https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive
* https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda

## 注意事项

* 本项目不需要`Conda`环境，但是鉴于需要学习使用其他项目，所以还是使用`Conda`区分不同环境。
