# FFmpeg

音视频框架

## 部署

```
# Linux
sudo apt install ffmpeg

# Windows
vcpkg install ffmpeg:x64-windows
```

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
