# FFmpeg

媒体处理工具

## 部署

```
# Linux
sudo apt install ffmpeg

# Windows
vcpkg install ffmpeg:x64-windows
```

#### Linux源码安装

```
# 目录
mkdir -p /data/ffmpeg ; cd $_

# nasm
cd /data/ffmpeg
wget https://www.nasm.us/pub/nasm/releasebuilds/2.16/nasm-2.16.tar.gz
tar -zxvf nasm-2.16.tar.gz
cd nasm-2.16/
./configure
sudo make -j 8 && sudo make install

# yasm
cd /data/ffmpeg
wget https://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
tar -zxvf yasm-1.3.0.tar.gz
cd yasm-1.3.0/
./configure
sudo make -j 8 && sudo make install

# libvpx?  --enable-gpl --enable-libvpx
cd /data/ffmpeg
#git clone https://chromium.googlesource.com/webm/libvpx.git
git clone https://github.com/webmproject/libvpx.git
cd libvpx/
git checkout v1.13.0
./configure --enable-static --enable-shared --enable-vp8 --enable-vp9 --enable-vp9-highbitdepth --as=yasm --disable-examples --disable-unit-tests
sudo make -j 8 && sudo make install

# libopus? --enable-gpl --enable-libopus
cd /data/ffmpeg
wget https://archive.mozilla.org/pub/opus/opus-1.3.1.tar.gz
tar -zxvf opus-1.3.1.tar.gz
cd opus-1.3.1/
./configure --enable-static --enable-shared
sudo make -j 8 && sudo make install

# libx264? --enable-gpl --enable-libx264
cd /data/ffmpeg
git clone https://code.videolan.org/videolan/x264.git
cd x264/
./configure --enable-static --enable-shared
sudo make -j 8 && sudo make install

# libx265? --enable-gpl --enable-libx265
cd /data/ffmpeg
git clone https://bitbucket.org/multicoreware/x265_git
cd x265_git/
git checkout 3.5
cd build/linux/
cmake -G "Unix Makefiles" ../../source/
sudo make -j 8 && sudo make install

# libopenh264? --enable-gpl --enable-libopenh264
cd /data/ffmpeg
git clone https://github.com/cisco/openh264.git
cd openh264/
git checkout v2.4.1
sudo make -j 8 && sudo make install

# ffmpeg
cd /data/ffmpeg
wget http://www.ffmpeg.org/releases/ffmpeg-6.1.1.tar.xz
tar -Jxvf ffmpeg-6.1.1.tar.xz
cd ffmpeg-6.1.1/
PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/"
./configure          \
--enable-static      \
--enable-shared      \
--enable-gpl         \
--enable-libvpx      \
--enable-libopus     \
--enable-libx264     \
--enable-libx265     \
--enable-libopenh264 \
--enable-encoder=libvpx_vp8 --enable-decoder=vp8 --enable-parser=vp8 \
--enable-encoder=libvpx_vp9 --enable-decoder=vp9 --enable-parser=vp9
sudo make -j 8 && sudo make install

# 链接
sudo vim /etc/ld.so.conf

---
/usr/local/lib/
---

ldconfig

# 验证
ffmpeg -version
ffmpeg -decoders
ffmpeg -encoders
```

## 常用功能

```
ffmpeg -ac 1 -ar 48000 -f s16le -i ./audio.pcm
ffplay -ac 1 -ar 48000 -f s16le -i ./audio.pcm

ffmpeg -i ./source.mp4 -vn -c:a mp3 -b:a 128k audio.mp3

ffmpeg -i ./source.mp3     -ac 1 -ar 48000 -f s16le audio.pcm
ffmpeg -i ./source.mp4 -vn -ac 1 -ar 48000 -f s16le audio.pcm

ffmpeg -ac 1 -ar 48000 -f s16le -i ./source.pcm -c:a mp3 -b:a 128k audio.mp3

ffmpeg -ac 1 -ar 48000 -f s16le -ss 10 -t  60 -i ./source.pcm -ac 1 -ar 48000 -f s16le audio.pcm
ffmpeg -ac 1 -ar 48000 -f s16le -ss 10 -to 70 -i ./source.pcm -ac 1 -ar 48000 -f s16le audio.pcm

$name="name"
ffmpeg -i .\$name.mp4 -vn -ac 1 -ar 48000 -c:a mp3 -b:a 128k .\$name.mp3
```

## 相关链接

## 注意事项
