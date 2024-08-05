# ollama

## 部署

```
# 目录
mkdir -p /data/ollama ; cd $_

# 环境
conda create --name ollama python=3.11
conda activate ollama

# 下载
# curl -fsSL https://ollama.com/install.sh | sh
wget https://ollama.com/install.sh

# 修改下载地址
vim install.sh

---
https://github.com/ollama/ollama/releases/download/v0.3.3/ollama-linux-amd64
---

# 安装
sh install.sh

# 验证
ollama -v

# 退出
conda deactivate
```

## 常用功能

```
# 
ollama ps
ollama list
ollama run
ollama pull
```

## 相关链接

* https://ollama.com
* https://github.com/ollama/ollama

## 注意事项

* 如果网络没有问题可以不用修改下载地址直接下载执行即可
