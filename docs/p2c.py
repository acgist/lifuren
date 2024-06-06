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

import torchvision
import torchvision.models   as models
import torchvision.datasets as datasets

# 默认参数
type         = "model"
device       = "cpu"
inputFile    = ""
inputFormat  = "none"
outputFile   = ""
outputFormat = "trace"

# 解析参数
opts, args = getopt.getopt(sys.argv[1:], "mdi:o:h", [
  "cpu", "gpu", "cuda",
  "model",
  "dataset",
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
    type = "model"
  if key == "-d" or key == "--dataset":
    type = "dataset"
  if key == "-i" or key == "--input":
    inputFile = value
  if key == "--input-format":
    inputFormat = value
  if key == "-o" or key == "--output":
    outputFile = value
  if key == "--output-format":
    outputFormat = value
  if key == "-h" or key == "--help":
    print(f"""

opts: {opts}
args: {args}

py2lib.py
--cpu|--gpu|--cuda
-m/--model
-d/--dataset
-i/--input=/data/input.th
--input-format=none|copy|state_dict
-o/--output=/data/output.pt
---output-format=copy|onnx|trace|script|state_dict
-h/--help

    """)
    sys.exit(0)

######## 处理模型

if type == "model":
  print("设备类型: ", device)
  print("输入文件: ", inputFile)
  print("输入文件格式: ", inputFormat)
  print("输出文件: ", outputFile)
  print("输出文件格式: ", outputFormat)
  model       = None
  modelName   = None
  weightsName = None
  checkLoop   = True
  if inputFormat != "copy":
    while checkLoop:
      print("支持模型：", end = "")
      for v in models.list_models():
        if modelName == None:
          print(v, end = " ")
        elif modelName in v:
          print(v, end = " ")
      print("")
      modelName = input("选择模型：")
      for v in models.list_models():
        if v == modelName:
          checkLoop = False
          break
      if modelName == "q":
        sys.exit(0)
  if inputFormat == "none":
    checkLoop = True
    while checkLoop:
      print("支持权重：None DEFAULT ", end = "")
      for v in models.get_model_weights(modelName):
        if weightsName == None:
          print(v.name, end = " ")
        elif weightsName in v.name:
          print(v.name, end = " ")
      print("")
      weightsName = input("选择权重：")
      for v in models.get_model_weights(modelName):
        if v.name == weightsName:
          print("输入大小", v.meta["min_size"])
          print("输出类型", v.meta["categories"])
          checkLoop = False
          break
      if weightsName == "None":
        checkLoop   = False
        weightsName = None
      if weightsName == "DEFAULT":
        checkLoop = False
      if weightsName == "q":
        sys.exit(0)
  
  # 加载模型
  if inputFormat == "none":
    model = models.get_model(modelName, weights = weightsName)
  if inputFormat == "copy":
    model = torch.load(inputFile)
  if inputFormat == "state_dict":
    model = models.get_model(modelName, weights = None)
    model.load_state_dict(torch.load(inputFile))
  if model == None:
    print("模型加载失败：", modelName, weightsName)
    sys.exit(0)

  # 模型定义
  for name, param in model.named_parameters():
    print({
        'tensor'      : name,
        'shape'       : list(param.size()),
        'trainable'   : param.requires_grad,
        'params_count': param.numel(),
    })

  # 评估模式
  model.eval()

  # 输出文件
  if len(outputFile) <= 0:
    os.makedirs("./model/" + outputFormat + "/", exist_ok = True)
    outputFile = "./model/" + outputFormat + "/" + modelName + "." + weightsName + ".pt"

  print("导出模型: ", modelName, weightsName)

  # 转换模型
  if outputFormat == "copy":
    # 包含模型结构以及权重
    torch.save(model, outputFile)
    # model = torch.load(inputFile)
  if outputFormat == "onnx":
    # onnx
    torch.onnx.export(model, torch.ones(1, 3, 224, 224).to(device), outputFile)
  if outputFormat == "trace":
    # TorchScript trace: 不能含有判断
    model = torch.jit.trace(model, torch.ones(1, 3, 224, 224).to(device))
    model.save(outputFile)
  if outputFormat == "script":
    # TorchScript script: 可以含有判断
    model = torch.jit.script(model)
    model.save(outputFile)
  if outputFormat == "state_dict":
    # 没有模型结构只有模型权重
    torch.save(model.state_dict(), outputFile)
    # model.load_state_dict(torch.load(inputFile))

######## 处理数据集

if type == "dataset":
  if len(outputFile) <= 0:
    os.makedirs("./dataset/", exist_ok = True)
    outputFile = "./dataset/"
  print("输出文件: ", outputFile)
  checkLoop   = True
  datasetName = None
  while checkLoop:
    print("支持的数据集：", end = "")
    for v in vars(datasets)["__all__"]:
      if datasetName == None:
        print(v, end = " ")
      elif datasetName in v:
        print(v, end = " ")
    print("")
    datasetName = input("选择数据集：")
    for v in vars(datasets)["__all__"]:
      if v == datasetName:
        checkLoop = False
        break
    if datasetName == "q":
      sys.exit(0)
  print("导出数据集: ", datasetName)
  dataset = getattr(datasets, datasetName)
  dataset(root = outputFile, train = True, download = True)

print("执行完成")
