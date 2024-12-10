# CLI API

## 命令

```
./lifuren[.exe] 命令 [参数...]
./lifuren[.exe] act          [act-tangxianzu  |act-guanhanqing  ] [train|pred] [model video_file|dataset model_name]
./lifuren[.exe] paint        [paint-wudaozi   |paint-gukaizhi   ] [train|pred] [model image_file|dataset model_name]
./lifuren[.exe] compose      [compose-shikuang|compose-liguinian] [train|pred] [model audio_file|dataset model_name]
./lifuren[.exe] poetize      [poetize-lidu    |poetize-suxin    ] [train|pred] [model rhythm prompt1 prompt2|dataset model_name]
./lifuren[.exe] pcm          dataset
./lifuren[.exe] pepper       dataset
./lifuren[.exe] embedding    [faiss|elasticsearch] dataset [pepper|ollama]
./lifuren[.exe] transform    [act-tangxianzu|act-guanhanqing...] [ONNX | TorchScript]
./lifuren[.exe] quantization [act-tangxianzu|act-guanhanqing...] model_path
./lifuren[.exe] [?|help]
```
