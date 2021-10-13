import torch
from torch import nn
from torchvision.models import alexnet, resnext101_32x8d

# 核心函数，参考了torch.quantization.fuse_modules()的实现
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    print(tokens)
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        print("--------",s)
        cur_mod = getattr(cur_mod, s)
        print(cur_mod)
    setattr(cur_mod, tokens[-1], module)

# 以AlexNet为例子
model = resnext101_32x8d(pretrained=False)

# 打印原模型
print("原模型")
# print(model)

# 打印每个层的名字，和当前配置
# 从而知道要改的层的名字
# for n, module in model.named_modules():
#     print(n, module, "\n")

# 假设要换掉AlexNet前2个卷积层，将通道从64改成128，其余参数不变
# 定义新层
layer0 = nn.Conv2d(512, 888, kernel_size=(1, 1), stride=(2, 2), bias=False)
layer1 = nn.BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
layer2 = nn.Conv2d(888, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
# 层的名字从上面19-20行的打印内容知道AlexNet前2个层的名字为 "features.0" 和 "features.3"
_set_module(model, 'layer3.0.downsample.0', layer0)
_set_module(model, 'layer3.0.downsample.1', layer1)
_set_module(model, 'layer3.1.conv1', layer2)

# 打印修改后的模型
print("新模型")
# print(model)

# 推理试一下
img = torch.rand((1, 3, 224, 224))
model(img)