# 模型工具

## PyTorch

#### TorchScript

```
# onnx
model = ...;
input = torch.randn(1, 3, 224, 224, device='cpu');
torch.onnx.export(model.eval(), input, "model.onnx", export_params=True);

# trace
model = torch.jit.trace(model.eval(), torch.zeros(3, 2));
model.save("trace.pt");

# script
model = torch.jit.script(model.eval());
model.save("script.pt");
```

> 需要PyTorch环境

## ONNX

通用模型

## Netron

可视化的工具

## TensorRT

计算加速
