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

## 依赖

|依赖|版本|官网|
|:--|:--|:--|
|fltk|1.3.8|https://github.com/fltk/fltk|
|json|3.11.2|https://github.com/nlohmann/json|
|cmake|3.26.4|https://github.com/Kitware/CMake|
|spdlog|1.12.0|https://github.com/gabime/spdlog|
|OpenCV|4.10.0|https://github.com/opencv/opencv|
|LibTorch|2.2.1|https://github.com/pytorch/pytorch|
|yaml-cpp|0.8.0|https://github.com/jbeder/yaml-cpp|
|llama.cpp|master|https://github.com/ggerganov/llama.cpp|
|cpp-httplib|0.15.3|https://github.com/yhirose/cpp-httplib|
|stable-diffusion.cpp|master|https://github.com/leejet/stable-diffusion.cpp|

## 模块

|模块|描述|详细描述|
|:--|:--|:--|
|boot|启动|`FLTK`界面、`REST`接口|
|core|核心|项目配置、模型加载、模型推理、模型微调、数据标记|
|deps|依赖|依赖项目|
|docs|文档|项目文档、使用说明|
|script|脚本|训练脚本、微调脚本|

## 文档

* [学习资料](./docs/AI.md)
* [部署文档](./docs/Deploy.md)
* [接口文档](./docs/REST.md)
* [功能说明](./docs/TODO.md)
