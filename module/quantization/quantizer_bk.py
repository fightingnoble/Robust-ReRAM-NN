from functools import wraps

import torch
import torch.nn as nn
from torch import Tensor

from .quant_functions import Round_STE, IntervalQuantizeIntO, IntervalQuantize
round_STE = Round_STE.apply
interval_quantize_int_o = IntervalQuantizeIntO.apply
interval_quantize = IntervalQuantize.apply

minimal_num = 1e-12
__all__ = ['Quantizer']


from .quant_functions import SymmSignedQuantize

symmsignedquantize = SymmSignedQuantize.apply


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calibration = False
        self.calibrated = False


class AverageLinearSignSymmIntervalQuantizer(Quantizer):
    def __init__(self, bitwidth=8):
        super().__init__()
        # self.bitwidth = bitwidth
        self.register_buffer('sum', torch.zeros(1,))
        self.register_buffer('counter', torch.zeros(1,))
        self.max_value = 2 ** (bitwidth - 1) - 1 
    
    def forward(self, tensor):
        with torch.no_grad():
            if self.training or self.calibration:
                interval = tensor.abs().max() / (self.max_value) + minimal_num
                self.counter.data += 1
                self.sum += interval
            else:
                interval = self.sum / self.counter

        tensor_quant = interval_quantize(tensor, interval, self.max_value, -self.max_value)
        return tensor_quant


class AverageLinearSignSymmIntervalQuantizerIntO(AverageLinearSignSymmIntervalQuantizer): 
    def forward(self, tensor):
        with torch.no_grad():
            if self.training or self.calibration:
                interval = tensor.abs().max() / (self.max_value) + minimal_num
                self.counter.data += 1
                self.sum += interval
            else:
                interval = self.sum / self.counter

        tensor_int = interval_quantize_int_o(tensor, interval, self.max_value, -self.max_value)
        return tensor_int


class AverageIntervalFunc(Quantizer):
    def __init__(self, bitwidth=8):
        super().__init__()
        # print('Executing Decorator\'s __init__() method')
        self.register_buffer('sum', torch.zeros(1,))
        self.register_buffer('counter', torch.zeros(1,))
        self.max_value = 2 ** (bitwidth - 1) - 1 
        
    def __call__(self, f):
        # print('Executing Decorator\'s __call__() method')
        @wraps(f)
        def wrap(tensor, *args, **kwargs):
            # print('Executing wrap()')
            with torch.no_grad():
                if self.training:
                    interval = tensor.abs().max() / (self.max_value) + minimal_num
                    self.counter.data += 1
                    self.sum += interval
                else:
                    interval = self.sum / self.counter
            # print('Decorator Parameters：', self.sum, self.counter)
            # print('Execute' + f.__name__ + '()')
            qa = f(tensor, interval, self.max_value, -self.max_value, *args, **kwargs)
            # print(f.__name__ + '() finished')
            return qa
        return wrap
    
@AverageIntervalFunc(8)
def average_linear_sign_symm_interval_quantizer(input, interval, max_int, min_int):
    return interval_quantize(input, interval, max_int, min_int)
    
@AverageIntervalFunc(8)
def average_linear_sign_symm_interval_quantizer(input, interval, max_int, min_int):
    return interval_quantize(input, interval, max_int, min_int)


if __name__=="__main__":
    print('Prepare to use decorated function')
    a = torch.rand(100)
    qa = average_linear_sign_symm_interval_quantizer(a)
    print('Test finished')
    print((a-qa).abs().max())