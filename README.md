# 李夫人

```
北方有佳人，绝世而独立。
一顾倾人城，再顾倾人国。
宁不知倾城与倾国，佳人难再得。
```

**学习`AI`相关知识，搞点东西练手，主要功能：图片识谱、指法标记、音频风格迁移。**

> 如果需要编辑谱面建议使用`musescore`专业编辑软件💃💃💃

----

<p align="center">
    <a target="_blank" href="https://starchart.cc/acgist/lifuren">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/acgist/lifuren?style=flat-square&label=Github%20stars&color=crimson" />
    </a>
    <img alt="Gitee stars" src="https://img.shields.io/badge/dynamic/json?style=flat-square&label=Gitee%20stars&color=crimson&url=https://gitee.com/api/v5/repos/acgist/lifuren&query=$.stargazers_count&cacheSeconds=3600" />
    <br />
    <img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/acgist/lifuren/build.yml?style=flat-square&branch=master" />
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/acgist/lifuren?style=flat-square&color=orange" />
    <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/acgist/lifuren?style=flat-square&color=blue" />
    <img alt="GitHub" src="https://img.shields.io/github/license/acgist/lifuren?style=flat-square&color=blue" />
</p>

## 模块

|模块|名称|详细描述|
|:--|:--|:--|
|docs|项目文档|项目文档|
|deps|依赖项目|依赖项目|
|boot|基础模块|项目配置、基础工具|
|core|核心模块|模型训练、模型推理|
|client|接口模块|`CLI`接口、`GUI`接口|

## 结构

```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |                         client                        |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |           CLI             |           GUI             |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |                          core                         |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| C/C++ |           audio           |           image           |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |      ffmpeg     |      libtorch     |      opencv     |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |                          boot                         |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |      spdlog     |      yaml-cpp     |     tinyxml2    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

## 文档

* [学习资料](./docs/AI.md)
* [命令文档](./docs/CLI.md)
* [计划文档](./docs/TODO.md)
* [模型文档](./docs/Model.md)
* [部署文档](./docs/Deploy.md)

## 主要功能

|任务|当前状态|详细描述|
|:--|:--:|:--|
|肖邦|○|五线谱识谱|
|师旷|○|音频风格迁移|
|莫扎特|○|钢琴指法标记|
|乐谱显示|○|简谱&五线谱|
|钢琴键盘|√||
|钢琴演奏|√||
|简谱移调|○||
|`CLI`接口|√|命令行接口|
|`GUI`接口|√|图形化接口|
|模型部署|○|`TensorRT`|
|模型部署|○|`OnnxRuntime`|

*√ - 完成、○ - 实现、# - 等待、? - 待定、~ - 忽略*

**因为一般用简谱的用户大部分不会需要用到五线谱，所以不会实现简谱识谱功能，而且功能实现和五线谱差不多做起来没啥动力。**

## 界面

![主界面](./docs/main.png)
![谱界面](./docs/score.png)
