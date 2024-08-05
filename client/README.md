# 服务终端

调用其他服务终端，设计参考[`SpringAI`](https://github.com/spring-projects/spring-ai)。

## 基础环境

```
sudo apt install build-essential
sudo apt install python3 python3-pip

# 配置
vim ~/.pip/pip.conf

---
[global]
index-url=https://pypi.tuna.tsinghua.edu.cn/simple
---

# 验证
pip3 config list
```

## PIP

```
pip install xxx
pip list
```