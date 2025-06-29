# FFmpeg

音视频框架

## 部署

```
# Linux
sudo apt install ffmpeg

# Windows
vcpkg install ffmpeg:x64-windows

# 源码编译
apt install nasm yasm
apt install libx264-dev libopenh264-dev
git clone https://github.com/FFmpeg/nv-codec-headers.git
cd nv-codec-headers
git switch sdk/12.1
sudo make install
wget http://www.ffmpeg.org/releases/ffmpeg-6.1.1.tar.xz
tar -Jxvf ffmpeg-6.1.1.tar.xz
cd ffmpeg-6.1.1/
PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/"
./configure            \
  --enable-static      \
  --enable-shared      \
  --enable-gpl         \
  --enable-libx264     \
  --enable-libopenh264 \
  --enable-cuda        \
  --enable-cuvid       \
  --enable-nvenc       \
  --enable-libnpp      \
  --enable-nonfree     \
  --enable-cuda-nvcc   \
  --extra-cflags="-I/usr/local/cuda/include" --extra-ldflags="-L/usr/local/cuda/lib64"
make -j
sudo make install
```

* 查看编码器和解码器`./configure --list-encoders ./configure --list-decoders`

## 常用功能

```
ffmpeg -decoders
ffmpeg -encoders

ffmpeg -ac 1 -ar 48000 -f s16le -i ./audio.pcm
ffplay -ac 1 -ar 48000 -f s16le -i ./audio.pcm

ffmpeg -i ./source.mp4 -vn -ac 1 -ar 48000 audio.mp3

ffmpeg -i ./source.mp3     -ac 1 -ar 48000 -f s16le audio.pcm
ffmpeg -i ./source.mp4 -vn -ac 1 -ar 48000 -f s16le audio.pcm

ffmpeg -ac 1 -ar 48000 -f s16le -i ./source.pcm -ac 1 -ar 48000 audio.mp3

ffmpeg -ac 1 -ar 48000 -f s16le -ss 10 -t  60 -i ./source.pcm -ac 1 -ar 48000 -f s16le audio.pcm
ffmpeg -ac 1 -ar 48000 -f s16le -ss 10 -to 70 -i ./source.pcm -ac 1 -ar 48000 -f s16le audio.pcm
```

## 相关链接

* https://ffmpeg.org/
* https://ffmpeg.org/ffmpeg-formats.html

## 注意事项

* 编码器和解码器可以按需安装
