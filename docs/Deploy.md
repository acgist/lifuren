# 部署

## Linux

#### 编译环境

```
sudo apt install cmake build-essential

sudo apt install gcc-11 g++-11
sudo apt install gcc-12 g++-12

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12

sudo update-alternatives --list       gcc
sudo update-alternatives --config     gcc
sudo update-alternatives --display    gcc
sudo update-alternatives --remove-all gcc
```

#### GCC-14

```
sudo apt install libmpc-dev libgmp-dev libmpfr-dev

# wget http://ftp.gnu.org/gnu/gcc/gcc-14.2.0/gcc-14.2.0.tar.gz
wget https://mirrors.aliyun.com/gnu/gcc/gcc-14.2.0/gcc-14.2.0.tar.gz
tar -zxvf gcc-14.2.0.tar.gz
cd gcc-14.2.0

./configure -v --prefix=/usr/local/gcc-14.2.0 --disable-multilib --enable-checking=release --enable-languages=c,c++
make -j
sudo make install

sudo ln -sf /usr/local/gcc-14.2.0/lib64/libstdc++.so.6.0.33 /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33
sudo ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33   /usr/lib/x86_64-linux-gnu/libstdc++.so.6

sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc-14.2.0/bin/gcc-14.2.0 14
sudo update-alternatives --install /usr/bin/g++ g++ /usr/local/gcc-14.2.0/bin/g++-14.2.0 14
```

#### 依赖下载

[build.yml](../.github/workflows/build.yml)

#### 编译命令

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

## 连接文件

```
vim /etc/ld.so.conf

---
/usr/local/lib/
/usr/local/lib64/
---

ldconfig
```

## Windows

#### 编译环境

* https://cmake.org/download/
* https://vcpkg.io/en/index.html
* https://visualstudio.microsoft.com/zh-hans/downloads/

#### 依赖下载

> `OpenCV`/`LibTorch`推荐使用官网下载（编译太慢）

* https://opencv.org/
* https://pytorch.org/

```
# 配置环境
VCPKG_DEFAULT_TRIPLET=x64-windows

# 安装依赖
vcpkg install opencv:x64-windows
vcpkg install spdlog:x64-windows
vcpkg install libtorch:x64-windows
vcpkg install wxwidgets:x64-windows

# 导出依赖
vcpkg export opencv    --zip
vcpkg export spdlog    --zip
vcpkg export libtorch  --zip
vcpkg export wxwidgets --zip
```

#### 编译命令

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

## 资源下载

[百度网盘](https://pan.baidu.com/s/1mNAXgaBV6lTQ1qkeFtnOtA?pwd=33p1)
