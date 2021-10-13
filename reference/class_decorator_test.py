from functools import wraps

import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor


class IntervalQuantize(Function):
    """
    Quantization with integer output and interval
    Backward using STE. 
    """
    @staticmethod
    def forward(ctx, input, interval, max_int, min_int):
        integer=torch.round_(input/interval).clamp_(min_int,max_int)
        out=integer*interval
        out.integer=integer
        out.interval=interval
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

interval_quantize = IntervalQuantize.apply

minimal_num = 1e-6


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calibration = False
        self.calibrated = None

    # def quantize(self, input, interval, max_int, min_int, *args, **kwargs):
    #     return input


class AverageInterval(Quantizer):
    def __init__(self, bitwidth=8):
        super().__init__()
        # self.bitwidth = bitwidth
        self.register_buffer('sum', torch.zeros(1,))
        self.register_buffer('counter', torch.zeros(1,))
        self.max_value = 2 ** (bitwidth - 1) - 1 

    def avg(self, tensor):
        with torch.no_grad():
            if self.training:
                interval = tensor.abs().max() / (self.max_value) + minimal_num
                self.counter.data += 1
                self.sum += interval
            else:
                interval = self.sum / self.counter
        return interval


def AverageIntervalModule(m):
    print('Executing Decorator')
    orig_forward = m.forward
    def wrap(self, tensor, *args, **kwargs):
        print('Executing wrap()')
        interval = self.avg(tensor)
        print('Decorator Parameters：', self.sum, self.counter)
        print('Execute' + orig_forward.__name__ + '()')
        qa = orig_forward(tensor, interval, self.max_value, -self.max_value, *args, **kwargs)
        print(orig_forward.__name__ + '() finished')
        return qa
    m.forward = wrap
    return m
  

@AverageIntervalModule
class AverageLinearSignSymmIntervalQuantizer(AverageInterval):
    forward = interval_quantize
    
print('装饰完毕')

quantizer = AverageLinearSignSymmIntervalQuantizer(8)

print('准备调用example()')
for i in range(15):
    a = torch.rand(100)
    qa = quantizer(a)

print('测试代码执行完毕')
print((a-qa).abs().max())