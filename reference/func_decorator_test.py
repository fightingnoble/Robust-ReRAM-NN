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

class Decorator(nn.Module):
    def __init__(self, bitwidth=8):
        super().__init__()
        print('执行类Decorator的__init__()方法')
        self.register_buffer('sum', torch.zeros(1,))
        self.register_buffer('counter', torch.zeros(1,))
        self.max_value = 2 ** (bitwidth - 1) - 1 
        
    def __call__(self, f):
        print('执行类Decorator的__call__()方法')
        def wrap(tensor, *args, **kwargs):
            print('执行wrap()')
            with torch.no_grad():
                if self.training:
                    interval = tensor.abs().max() / (self.max_value) + minimal_num
                    self.counter.data += 1
                    self.sum += interval
                else:
                    interval = self.sum / self.counter
            print('装饰器参数：', self.sum, self.counter)
            print('执行' + f.__name__ + '()')
            qa = f(tensor, interval, self.max_value, -self.max_value, *args, **kwargs)
            print(f.__name__ + '()执行完毕')
            return qa
        return wrap
    
@Decorator(8)
def AverageLinearSignSymmIntervalQuantizer(input, interval, max_int, min_int):
    return interval_quantize(input, interval, max_int, min_int)
    
print('装饰完毕')

print('准备调用example()')
for i in range(15):
    a = torch.rand(100)
    qa = AverageLinearSignSymmIntervalQuantizer(a)

print('测试代码执行完毕')
print((a-qa).abs().max())