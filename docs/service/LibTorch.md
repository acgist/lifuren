# LibTorch

机器学习框架

## 部署

```
# Linux
sudo apt install libtorch-dev

# Windows
vcpkg install libtorch:x64-windows
```

## 相关链接

* https://pytorch.org/
* https://pytorch.org/get-started/locally/
* https://download.pytorch.org/libtorch/cpu/
* https://download.pytorch.org/libtorch/cu126/

## 注意事项

* 建议直接使用官网下载避免编译时间过长
* 警告`Could NOT find nvtx3 (missing: nvtx3_dir)`添加编译参数`-DUSE_SYSTEM_NVTX=ON`
* 警告`CMake Warning (dev) at /usr/share/cmake-3.22/Modules/FindPackageHandleStandardArgs.cmake:438`添加编译参数`-Wno-dev`
* 警告`CMake Warning at deps/libtorch/share/cmake/Caffe2/public/cuda.cmake:140`修改对应位置代码`Python::Interpreter -> ${Python_EXECUTABLE}`
