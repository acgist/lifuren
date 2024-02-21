# 部署

## 源码编译

### Linux

```
mkdir build
cd build
cmake ..
make
# make install
# export LD_LIBRARY_PATH=/data/dev/lifuren/target/lib/:/data/dev/lifuren/deps/libtorch/lib/:$LD_LIBRARY_PATH
```

### Windows

```
mkdir build
cd build
cmake -G "Visual Studio 17 2022" ..
cmake --build . --config Debug
# cmake --install . --config Debug
```

## 依赖下载

```
# Linux安装依赖

sudo apt install libmlpack-dev
sudo apt install libopencv-dev
sudo apt install libfltk1.3-dev

# Windows配置环境变量

VCPKG_DEFAULT_TRIPLET=x64-windows

# Windows安装依赖

vcpkg install fltk
vcpkg install mlpack

# Windows导出依赖

vcpkg export fltk   --zip
vcpkg export mlpack --zip
```

> `Windows`开发时`OpenCV`和`LibTorch`直接官网下载

* https://opencv.org/releases/
* https://pytorch.org/get-started/locally/

## 开发环境

* https://code.visualstudio.com/
* https://visualstudio.microsoft.com/zh-hans/downloads/

## LibTorch

```
# https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip
# https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
# https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.0.1%2Bcu118.zip
# https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
# https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.0.1%2Bcpu.zip
# https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.0.1%2Bcpu.zip
# https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.0.1%2Bcu118.zip
# https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-debug-2.0.1%2Bcu118.zip
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
# https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe
# https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
# https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
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
```

> 不用单独安装`CUDA`已经包含

## 相关下载

```
https://pan.baidu.com/s/1mNAXgaBV6lTQ1qkeFtnOtA?pwd=33p1
```
