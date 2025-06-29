# OpenCV

机器视觉框架

## 部署

```
# Linux
sudo apt install libopencv-dev

# Windows
vcpkg install opencv:x64-windows

# 源码编译
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.10.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
cd opencv-4.10.0
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -WITH_FFMPEG=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DOPENCV_GENERATE_PKGCONFIG=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.10.0/modules/
make -j
sudo make install
```

## 相关链接

* https://opencv.org/
* https://opencv.org/releases/

## 注意事项

* 避免使用源码安装编译时间过长
