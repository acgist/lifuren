# 部署

## 源码编译

#### Linux

###### 编译环境

```
sudo apt install build-essential

sudo apt install gcc-11 g++-11
sudo apt install gcc-12 g++-12

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12

sudo update-alternatives --config  gcc
sudo update-alternatives --display gcc

apt install --reinstall --print-uris libfaiss-dev
```

###### 依赖下载

[build.yml](../.github/workflows/build.yml)

###### 编译命令

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug|Release ..
make -j
make install

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug|Release ..
cmake --build . -j
cmake --build . --parallel 8
cmake --install .
```

#### Windows

#### 编译环境

* https://visualstudio.microsoft.com/zh-hans/downloads/

###### 依赖下载

```
# 配置环境
VCPKG_DEFAULT_TRIPLET=x64-windows

# 安装依赖
vcpkg install fltk:x64-windows
vcpkg install faiss:x64-windows
vcpkg install opencv:x64-windows
vcpkg install spdlog:x64-windows
vcpkg install yaml-cpp:x64-windows
vcpkg install cpp-httplib:x64-windows

# 导出依赖
vcpkg export fltk        --zip
vcpkg export faiss       --zip
vcpkg export opencv      --zip
vcpkg export spdlog      --zip
vcpkg export yaml-cpp    --zip
vcpkg export cpp-httplib --zip
```

* `OpenCV`可以直接官网下载
* `LibTorch`可以直接官网下载

###### 编译命令

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug|Release -T host=x64 -A x64 -G "Visual Studio 17 2022" ..
cmake --config Debug|Release --build . -j
cmake --config Debug|Release --build . --parallel 8
cmake --install .
```

## 开发环境

* https://code.visualstudio.com/

## 模型

* https://hf-mirror.com/models
* https://huggingface.co/models

## 数据集

* https://hf-mirror.com/datasets
* https://huggingface.co/datasets

## 服务部署

[服务部署](./service)

## 资源下载

[百度网盘](https://pan.baidu.com/s/1mNAXgaBV6lTQ1qkeFtnOtA?pwd=33p1)
