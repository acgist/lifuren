# NVIDIA

## 部署

```
# 目录
mkdir -p /data/nvidia ; cd $_

# 驱动
sudo apt install nvidia-driver-560

# CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-12-6
sudo apt install cudnn9-cuda-12-6
sudo apt install cuda-toolkit-12-6

# 验证
nvcc -V
```

## 常用功能

```
nvidia-smi
```

## 相关链接

* https://www.nvidia.cn/Download/index.aspx?lang=cn

#### CUDA

* https://developer.nvidia.com/cuda-toolkit-archive

#### cuDNN

* https://developer.nvidia.cn/cudnn
* https://developer.nvidia.com/cudnn
* https://developer.nvidia.cn/cudnn-downloads
* https://developer.nvidia.com/cudnn-downloads
* https://docs.nvidia.com/deeplearning/cudnn/index.html

## 注意事项

* 根据实际情况选择版本
* `CUDA`已经包含`cuDNN`不用单独安装
