# 部署

## 依赖

|依赖|版本|官网|
|:--|:--|:--|
|fltk|1.3.8|https://github.com/fltk/fltk|
|json|3.11.2|https://github.com/nlohmann/json|
|cmake|3.26.4|https://github.com/Kitware/CMake|
|spdlog|1.12.0|https://github.com/gabime/spdlog|
|OpenCV|4.10.0|https://github.com/opencv/opencv|
|LibTorch|2.2.1|https://github.com/pytorch/pytorch|
|yaml-cpp|0.8.0|https://github.com/jbeder/yaml-cpp|
|cpp-httplib|0.15.3|https://github.com/yhirose/cpp-httplib|

## 服务

|服务|版本|官网|
|:--|:--|:--|
yolo
neo4j
ollama
openai
elasticsearch

## 源码编译

#### Linux

###### 编译环境

```
sudo apt install build-essential
```

###### 依赖下载

[build.yml](../.github/workflows/build.yml)

###### 编译命令

```
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Debug|Release ..
make
make install

mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Debug|Release ..
cmake --build . --parallel 8 --config Debug|Release
cmake --install .

export LD_LIBRARY_PATH=/data/dev/lifuren/install/lib/:/data/dev/lifuren/deps/libtorch/lib/:$LD_LIBRARY_PATH
```

#### Windows

#### 编译环境

* https://visualstudio.microsoft.com/zh-hans/downloads/

###### 依赖下载

```
# 配置环境
VCPKG_DEFAULT_TRIPLET=x64-windows

# 安装依赖
vcpkg install fltk
vcpkg install opencv
vcpkg install spdlog
vcpkg install libtorch
vcpkg install yaml-cpp
vcpkg install cpp-httplib

# 导出依赖
vcpkg export fltk        --zip
vcpkg export opencv      --zip
vcpkg export spdlog      --zip
vcpkg export libtorch    --zip
vcpkg export yaml-cpp    --zip
vcpkg export cpp-httplib --zip
```

> `Windows`开发`OpenCV`和`LibTorch`直接官网下载

* https://opencv.org/releases/
* https://pytorch.org/get-started/locally/

###### 编译命令

```
mkdir build
cd build
cmake -G "Visual Studio 17 2022" ..
cmake --build . --parallel 8 --config Debug|Release
cmake --install .
```

#### 注意事项

* `windows`需要删除`fltk/unofficial-brotli`模块

## 开发环境

* https://code.visualstudio.com/

#### NVIDIA

[Nvidia](./tutorial/Nvidia.md)

#### LibTorch

[Nvidia](./tutorial/LibTorch.md)

## 模型

* https://hf-mirror.com/models
* https://huggingface.co/models

## 数据集

* https://hf-mirror.com/datasets
* https://huggingface.co/datasets
* https://github.com/chinese-poetry/chinese-poetry
* https://github.com/chinese-poetry/chinese-poetry-zhCN

## 模型

#### 模型下载

* https://hf-mirror.com/models
* https://huggingface.co/models

#### 模型微调

[模型微调](./optimize/Finetune.md)

#### 模型量化

[模型量化](./optimize/Quantization.md)

#### 模型部署

[模型部署](./model)

#### 服务部署

[模型部署](./tutorial)

## 线上部署

一般来说如果租用线上`GPU`服务器非常贵，所以建议应用服务放到线上，`AI`相关服务本地部署使用`frp`/`autossh`内网穿透软件为线上应用提供服务。
