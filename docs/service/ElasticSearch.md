# ElasticSearch

`ES`提供向量存储搜索功能，推荐使用`Faiss`。

## 部署

```
# 目录
mkdir -p /data/elasticsearch ; cd $_

# 下载
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.7.1-linux-x86_64.tar.gz

# 用户
sudo useradd -s /sbin/nologin elasticsearch

# 安装
tar -zxvf elasticsearch-8.7.1-linux-x86_64.tar.gz
rm elasticsearch-8.7.1-linux-x86_64.tar.gz
cd elasticsearch-8.7.1
sudo chown -R elasticsearch:elasticsearch /data/elasticsearch

# 配置
sudo vim config/elasticsearch.yml

---
network.host: 0.0.0.0
discovery.type: single-node
xpack.security.enabled: true
---

sudo vim config/jvm.options

---
-Xms2g
-Xmx2g
---

sudo vim bin/elasticsearch-env

---
ES_JAVA_HOME=/data/elasticsearch/elasticsearch-8.7.1/jdk
---

# 服务
sudo vim /usr/lib/systemd/system/elasticsearch.service

---
[Unit]
Description=elasticsearch
After=network.service
Wants=network.service

[Service]
Type=forking
User=elasticsearch
ExecStart=/data/elasticsearch/elasticsearch-8.7.1/bin/elasticsearch -d
ExecReload=/bin/kill -HUP $MAINPID
KillMode=process
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target graphic.target
---

sudo systemctl daemon-reload
sudo systemctl start  elasticsearch.service
sudo systemctl enable elasticsearch.service

# 密码
sudo bin/elasticsearch-setup-passwords interactive

# 验证
curl http://elastic:elastic@localhost:9200
```

## 常用功能

```
http://localhost:9200
http://localhost:9200/index
http://localhost:9200/index/_search
```

## 相关链接

* https://www.elastic.co/cn
* https://github.com/elastic/elasticsearch

## 注意事项
