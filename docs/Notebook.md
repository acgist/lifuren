# 备忘

## Git常用命令

```
# 克隆仓库
git clone --branch branch --depth 1 https://github.com/git.git
# 更新模块
git submodule update --init --recursive --depth 1
# 新增模块
git submodule add -branch branch --depth 1 https://github.com/git.git deps/git
```

## Linux常用命令

```
sudo apt install libmgl-dev
sudo apt install libcv-dev
sudo apt install libopencv-dev
```

## Windows常用命令

```
vcpkg install mathgl
vcpkg install opencv

vcpkg install dlib
vcpkg install glog
vcpkg install nlohmann-json

vcpkg export mathgl --zip
```
