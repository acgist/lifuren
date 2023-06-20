# 编译

## linux

```
mkdir build
cd build
cmake ..
make
# make install
```

## windows

```
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
make
# make install
```

## NVIDA

```
CPU/GPU/CUDA/cuDNN
```

* https://www.nvidia.cn/Download/index.aspx?lang=cn

## CUDA

* https://developer.nvidia.com/cuda-toolkit-archive
* https://developer.nvidia.com/cuda-11-8-0-download-archive

```
# debian
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-debian11-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
# ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

## cuDNN

* https://developer.nvidia.com/zh-cn/cudnn
* https://developer.nvidia.com/rdp/cudnn-download
* https://docs.nvidia.com/deeplearning/cudnn/index.html
* https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
* https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html

```
# https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.2/local_installers/11.x/cudnn-windows-x86_64-8.9.2.26_cuda11-archive.zip/
# https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.2/local_installers/11.x/cudnn-linux-x86_64-8.9.2.26_cuda11-archive.tar.xz/
# https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.2/local_installers/11.x/cudnn-local-repo-debian11-8.9.2.26_1.0-1_amd64.deb/
# https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.2/local_installers/11.x/cudnn-local-repo-ubuntu2004-8.9.2.26_1.0-1_amd64.deb/

sudo apt-get install zlib1g

sudo dpkg -i cudnn-local-repo-${distro}-8.x.x.x_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8=8.x.x.x-1+cudaX.Y
sudo apt-get install libcudnn8-dev=8.x.x.x-1+cudaX.Y
sudo apt-get install libcudnn8-samples=8.x.x.x-1+cudaX.Y

sudo apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
sudo apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
sudo apt-get install libcudnn8-samples=${cudnn_version}-1+${cuda_version}
```
