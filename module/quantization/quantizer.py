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

    # def quantize(self, input, interval, max_int, min_int, *args, **kwargs):
    #     return input


class AverageInterval(Quantizer):
    def __init__(self, bitwidth=8):
        super().__init__()
        # self.bitwidth = bitwidth
        self.register_buffer('sum', torch.zeros(1,))
        self.register_buffer('counter', torch.zeros(1,))
        self.max_value = 2 ** (bitwidth - 1) - 1 

    def get_interval(self, tensor):
        with torch.no_grad():
            if self.training:
                interval = tensor.abs().max() / (self.max_value) + minimal_num
                self.counter.data += 1
                self.sum += interval
            else:
                interval = self.sum / self.counter
        return interval


def FuseRangeAndStep(m):
    # print('Executing Decorator')
    orig_forward = m.forward
    def wrap(self, tensor, *args, **kwargs):
        # print('Executing wrap()')
        interval = self.get_interval(tensor)
        # print('Decorator Parameters：', self.sum, self.counter)
        # print('Execute ' + orig_forward.__name__ + '()')
        qa = orig_forward(tensor, interval, self.max_value, -self.max_value, *args, **kwargs)
        # print(orig_forward.__name__ + '() finished')
        return qa
    m.forward = wrap
    return m
  

@FuseRangeAndStep
class AverageLinearSignSymmIntervalQuantizer(AverageInterval):
    forward = interval_quantize


@FuseRangeAndStep
class AverageLinearSignSymmIntervalQuantizerIntO(AverageInterval):
    forward = interval_quantize_int_o


class Maxinterval(Quantizer):
    def __init__(self, bitwidth=8) -> None:
        super().__init__()
        # self.bitwidth = bitwidth
        self.max_value = 2 ** (bitwidth - 1) - 1 

    def get_interval(self, x) -> Tensor:
        interval = x.abs().max() / self.max_value + minimal_num
        return interval


@FuseRangeAndStep
class UniformQ(Maxinterval):
    forward = interval_quantize



if __name__=="__main__":
    print('Prepare to use decorated function')

    quantizer = AverageLinearSignSymmIntervalQuantizer(8)

    for i in range(15):
        a = torch.rand(100)
        qa = quantizer(a)

    print('Test finished')
    print((a-qa).abs().max())
