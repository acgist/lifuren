# 模型工具

## PyTorch

#### TorchScript

```
# onnx
model = ...
torch.onnx.export(model.eval(), torch.randn(1, 3, 224, 224), "model.onnx")

# trace
model = torch.jit.trace(model.eval(), torch.randn(1, 3, 224, 224))
model.save("trace.pt")

# script
model = torch.jit.script(model.eval())
model.save("script.pt")
```

> 需要PyTorch环境

## ONNX

通用模型

## Netron

可视化的工具

## TensorRT

计算加速
