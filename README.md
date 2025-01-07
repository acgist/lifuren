# 李夫人

```
北方有佳人，绝世而独立。
一顾倾人城，再顾倾人国。
宁不知倾城与倾国，佳人难再得。
```

项目主要研发方向是生成网络、机器视觉、自然语言处理。

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
|deps|依赖项目|依赖项目|
|core|基础模块|项目配置、日志配置、基础工具|
|model|模型模块|学习框架、模型训练、模型推理、模型微调、模型量化|
|client|终端模块|依赖项目调用终端|
|cv|机器视觉|音频生成、图片生成、视频生成|
|nlp|自然语言|诗词解析、诗词嵌入、诗词生成|
|boot|启动模块|`CLI`接口、`GUI`接口、`REST`接口|
|docs|项目文档|项目文档|

## 结构

```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |                         boot                          |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |     CLI     |    REST     |          GUI/FLTK         |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |            nlp            |            cv             |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |           poetry          |   audio   image   video   |
| C/C++ +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |    ES/Faiss    ollama     |     ffmpeg     opencv     |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |          client           |           model           |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |          httplib          |          libtorch         |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |                         core                          |
|       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       |    spdlog     |    yaml-cpp     |    nlohmann_json    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

## 模型

|模型|类型|状态|描述|
|:--|:--|:--:|:--|
|李杜|诗词|○|AI作诗|
|苏辛|诗词|○|AI填词|
|师旷|音频|○|AI音频风格迁移|
|吴道子|视频|○|AI视频动作预测|

> *√=完成、○-进行中、#-未开始、?-待定、--忽略*

## 文档

* [学习资料](./docs/AI.md)
* [命令文档](./docs/CLI.md)
* [接口文档](./docs/REST.md)
* [计划文档](./docs/TODO.md)
* [模型文档](./docs/Model.md)
* [部署文档](./docs/Deploy.md)
