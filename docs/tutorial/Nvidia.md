# NVIDIA

## 部署

```
# 目录
mkdir -p /data/nvidia ; cd $_

# cuda
sudo apt install nvidia-cuda-toolkit

# 环境

# 下载

# 安装

# 配置

# 环境变量

# 验证

# 退出
```

## 常用功能

```
nvcc
nvidia-smi
```

## 相关链接

* https://www.nvidia.cn/Download/index.aspx?lang=cn

#### CUDA

* https://developer.nvidia.com/cuda-toolkit-archive
* https://developer.nvidia.com/cuda-11-8-0-download-archive

```
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe
https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
```

#### cuDNN

* https://developer.nvidia.com/zh-cn/cudnn
* https://developer.nvidia.com/rdp/cudnn-download
* https://docs.nvidia.com/deeplearning/cudnn/index.html
* https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
* https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html

```
https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.2/local_installers/11.x/cudnn-windows-x86_64-8.9.2.26_cuda11-archive.zip/
https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.2/local_installers/11.x/cudnn-linux-x86_64-8.9.2.26_cuda11-archive.tar.xz/
https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.2/local_installers/11.x/cudnn-local-repo-debian11-8.9.2.26_1.0-1_amd64.deb/
https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.2/local_installers/11.x/cudnn-local-repo-ubuntu2004-8.9.2.26_1.0-1_amd64.deb/
```

## 注意事项

* `CUDA`已经包含`cuDNN`不用单独安装
