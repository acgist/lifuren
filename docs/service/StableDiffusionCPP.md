# stable-diffusion.cpp

## 部署

```
# 目录
mkdir -p /data/stable-diffusion.cpp ; cd $_

# 下载
git clone https://github.com/leejet/stable-diffusion.cpp.git
cd stable-diffusion.cpp

# Linux安装
mkdir build ; cd $_
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Window安装
mkdir build ; cd $_
cmake ..
cmake --build . -j --config Release

# 验证
./sd -v
```

## 常用功能

```
./sd -m ./model.ckpt -p 'prompt' --steps 30
```

## 相关链接

* https://developer.nvidia.com/cuda-downloads
* https://github.com/leejet/stable-diffusion.cpp

## 注意事项

* 使用`CUDA`需要安装相关依赖，添加编译参数`-DSD_CUBLAS=ON`。
