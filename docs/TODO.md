# 计划

> √ - 完成、○ - 实现、# - 等待、? - 待定、~ - 忽略

## 主要功能

|任务|当前状态|详细功能|
|:--|:--:|:--|
|音频|○|师旷|
|视频|○|吴道子|

## 详细功能

|任务|当前状态|详细描述|
|:--|:--:|:--|
|师旷|○|音频风格迁移|
|吴道子|○|视频动作预测|
|音频嵌入|○|-|
|视频嵌入|~|视频无需嵌入|
|`CLI`接口|√|命令行接口|
|`GUI`接口|√|图形界面接口|
|`REST`接口|√|只有推理接口|
|模型部署|○|`ONNX`|
|模型部署|○|`TensorRT`|
|模型微调|?|-|
|模型量化|?|-|
|编译时间|?|优化编译时间|

## 代码规范

* 避免使用异常
* 必须使用命名空间
* 宏定义必须使用`LFR_`开头
* 优先使用智能指针（避免使用裸指针）
* 除了变量之外避免使用复数（方法、类名）
* 类名及其方法使用驼峰命名规则
* 全局变量和方法使用下划线命名规则
* 在可能出现溢出的地方使用`LL/ULL`后缀
* 头文件宏定义`LFR_HEADER_MODULE_PATH_FILENAME_HPP`

## GIT日志前缀

```
* = 修改代码
+ = 增加功能
- = 减少功能
~ = 优化代码
% = 重要更新
@ = 修复问题
! = 版本发布
$ = 配置更新
# = 文档更新
& = 依赖升级
? = 其他修改
```
