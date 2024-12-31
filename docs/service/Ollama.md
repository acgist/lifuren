# ollama

模型部署

## 部署

```
# 目录
mkdir -p /data/ollama ; cd $_

# 环境
conda create --name ollama python=3.12
conda activate ollama

# 下载
# curl -fsSL https://ollama.com/install.sh | sh
wget https://ollama.com/install.sh

# 修改下载地址
vim install.sh

---
https://github.com/ollama/ollama/releases/download/v0.3.3/ollama-linux-amd64
---

# 修改端口
sudo vim /etc/systemd/system/ollama.service

---
Environment="OLLAMA_HOST=0.0.0.0:11434"
---

# 重启服务
sudo systemctl daemon-reload
sudo systemctl restart ollama

# 安装
sh install.sh

# 验证
ollama -v

# 退出
conda deactivate
```

## 常用功能

```
# 查看运行模型
ollama ps
# 查看下载模型
ollama list
# 运行模型
ollama run 模型名称
# 下载模型
ollama pull 模型名称

# bge-large
ollama run bge-large

# quentinz/bge-large-zh-v1.5
ollama run quentinz/bge-large-zh-v1.5
```

## 相关链接

* https://ollama.com
* https://ollama.com/models
* https://github.com/ollama/ollama

## 注意事项

* 如果网络没有问题可以不用修改下载地址直接下载执行即可
