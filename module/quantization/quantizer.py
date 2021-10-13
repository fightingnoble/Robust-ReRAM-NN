from functools import wraps
from typing import Any, Callable

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


def _quant_unimplemented(self, *input: Any) -> None:
    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError


def _get_interval_unimplemented(self, *input: Any) -> None:
    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calibration = False
        self.calibrated = None

    # def quantize(self, input, interval, max_int, min_int, *args, **kwargs):
    #     return input

    quant: Callable[..., Any] = _quant_unimplemented
    get_interval: Callable[..., Any] = _get_interval_unimplemented

    def reset(self):
        self.calibrated = None


class IntervalQuantizer(Quantizer):
    def __init__(self, bitwidth=8):
        super().__init__()
        # self.bitwidth = bitwidth
        self.max_value = 2 ** (bitwidth - 1) - 1 

    def forward(self, tensor, *args, **kwargs):
        # print('Executing wrap()')
        interval = self.get_interval(tensor)
        # print('Decorator Parameters：', self.sum, self.counter)
        # print('Execute ' + self.quant.__name__ + '()')
        qa = self.quant(tensor, interval, self.max_value, -self.max_value, *args, **kwargs)
        # print(self.quant.__name__ + '() finished')
        return qa
    

class AverageInterval(IntervalQuantizer):
    def __init__(self, bitwidth=8):
        super().__init__(bitwidth)
        self.register_buffer('sum', torch.zeros(1,))
        self.register_buffer('counter', torch.zeros(1,))

    def get_interval(self, tensor):
        with torch.no_grad():
            # print((self.counter))
            if self.training or (self.calibration and not self.calibrated):
                interval = tensor.abs().max() / (self.max_value) + minimal_num
                self.counter.data += 1
                self.sum.data += interval
            else:
                interval = self.sum / self.counter
        return interval

    def reset(self):
        super().reset()
        self.register_buffer('sum', torch.zeros(1,))
        self.register_buffer('counter', torch.zeros(1,))

    def extra_repr(self) -> str:
        return super().extra_repr() + "id:{}".format(id(self))


class Maxinterval(IntervalQuantizer):
    def get_interval(self, x) -> Tensor:
        interval = x.abs().max() / self.max_value + minimal_num
        return interval


class AverageLinearSignSymmIntervalQuantizer(AverageInterval):
    quant = interval_quantize


class AverageLinearSignSymmIntervalQuantizerIntO(AverageInterval):
    quant = interval_quantize_int_o

class UniformQ(Maxinterval):
    quant = interval_quantize


if __name__=="__main__":
    print('Prepare to use decorated function')

    quantizer1 = AverageLinearSignSymmIntervalQuantizer(8)
    quantizer2 = AverageLinearSignSymmIntervalQuantizer(8)


    for i in range(15):
        a = torch.rand(100)
        qa = quantizer1(a)
        qa = quantizer2(a)
        # print(id(quantizer))
        print(id(quantizer1.sum))

    print('Test finished')
    print((a-qa).abs().max())
