#!/usr/bin/python

"""
# PyTorch模型转为LibTorch模型

pip|conda install onnx
pip|conda install torchvision

https://pytorch.org/vision/stable/models.html
https://pytorch.org/vision/stable/datasets.html
"""

import os
import sys
import getopt

import onnx
import torch

import torchvision.models   as models
import torchvision.datasets as datasets

# 默认参数
device="cpu"
modelName=""
modelWeights="IMAGENET1K_V1"
datasetName=""
input=""
inputFormat="none"
output=""
outputFormat="trace"

# 解析参数
opts, args = getopt.getopt(sys.argv[1:], "-m:-d:-i:-if:-o:-of:-h", [
  "cpu", "gpu", "cuda",
  "model=",
  "weights=",
  "dataset=",
  "input=",
  "input-format=",
  "output=",
  "output-format=",
  "help"
])

for key, value in opts:
  if key == "--cpu":
    device = "cpu"
  if key == "--gpu" or key == "--cuda":
    device = "cuda"
  if key == "-m" or key == "--model":
    modelName = value
  if key == "-w" or key == "--weights":
    modelWeights = value
  if key == "-d" or key == "--dataset":
    datasetName = value
  if key == "-i" or key == "--input":
    input = value
  if key == "-if" or key == "--input-format":
    inputFormat = value
  if key == "-o" or key == "--output":
    output = value
  if key == "-of" or key == "--output-format":
    outputFormat = value
  if key == "-h" or key == "--help":
    print(f"""

opts: {opts}
args: {args}

py2lib.py
--cpu|gpu|cuda
-m/--model=[vgg16_bn|resnet18|mobilenet_v2...]
-w/--weights=[IMAGENET1K_V1|IMAGENET1K_V1|IMAGENET1K_V1...]
-d/--dataset=[MNIST|CIFAR10...]
-i/--input=/data/source.th
--input-format=none|copy|state_dict
-o/--output=/data/target.pt
--output-format=none|copy|onnx|trace|script|state_dict
-h/--help

    """)
    sys.exit(0)

# 校验参数
if len(modelName) <= 0 and len(datasetName) <= 0:
  print("请指定模型名称或者数据集名称")
  sys.exit(0)

######## 处理模型

if len(modelName) > 0:
  if len(input) <= 0:
    input = "input.pt"
  if len(output) <= 0:
    os.mkdir("./model/")
    output = "./model/output.pt"
  print("设备类型: ", device)
  print("模型名称: ", modelName)
  print("模型权重: ", modelWeights)
  print("输入文件: ", input)
  print("输入文件格式: ", inputFormat)
  print("输出文件: ", output)
  print("输出文件格式: ", outputFormat)
  # 加载模型
  model = None
  if inputFormat == "copy" and len(input) > 0:
    model = torch.load(input)
  else:
    # 加载模型
    if modelName == "vgg16_bn":
      model = models.vgg16_bn(weights=modelWeights)
    if modelName == "resnet18":
      model = models.resnet18(weights=modelWeights)
    if modelName == "mobilenet_v2":
      model = models.mobilenet_v2(weights=modelWeights)
    # 加载权重
    if len(input) > 0:
      if inputFormat == "copy":
        model = torch.load(input)
      if inputFormat == "state_dict":
        model.load_state_dict(torch.load(input))

  # 评估模式
  model.eval()

  # 转换模型
  if outputFormat == "copy":
    # 包含模型结构以及权重
    torch.save(model, output)
    # model = torch.load(input)
  if outputFormat == "onnx":
    # onnx
    torch.onnx.export(model, torch.ones(1, 3, 224, 224).to(device), output)
  if outputFormat == "trace":
    # TorchScript trace: 不能含有判断
    model = torch.jit.trace(model, torch.ones(1, 3, 224, 224).to(device))
    model.save(output)
  if outputFormat == "script":
    # TorchScript script: 可以含有判断
    model = torch.jit.script(model)
    model.save(output)
  if outputFormat == "state_dict":
    # 没有模型结构只有模型权重
    torch.save(model.state_dict(), output)
    # model.load_state_dict(torch.load(path))

######## 处理数据集

if len(datasetName) > 0:
  if len(output) <= 0:
    os.mkdir("./dataset/")
    output = "./dataset/"
  print("数据集名称: ", datasetName)
  print("输出文件: ", output)
  if datasetName == "MNIST":
    datasets.MNIST(root=output, train=True, download=True)
  if datasetName == "CIFAR10":
    datasets.CIFAR10(root=output, train=True, download=True)

print("执行完成")
