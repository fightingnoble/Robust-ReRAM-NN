import math
from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Function


class Quant_F(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, bits, overflow_rate):
        # ctx is a context object that can be used to stash information for backward computation
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None


def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, torch.Tensor):
        v = float(v.data.item())
    sf = math.ceil(math.log2(v + 1e-12))
    return sf


def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits - 1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)
    # print(delta, min_val, max_val, bound)
    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value


def log_minmax_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = min_max_quantize(input0, bits - 1)
    v = torch.exp(v) * s
    return v


def log_linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = linear_quantize(input0, sf, bits - 1)
    v = torch.exp(v) * s
    return v


def min_max_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()

    if isinstance(min_val, torch.Tensor):
        max_val = float(max_val.data.item())
        min_val = float(min_val.data.item())

    input_rescale = (input - min_val) / (max_val - min_val)

    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = v * (max_val - min_val) + min_val
    return v


def tanh_quantize(input, bits, mode=0):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input = torch.tanh(input)  # [-1, 1]
    if mode==0:
        input_rescale = (input/input.max() + 1.0) / 2  # [0, 1]
        n = math.pow(2.0, bits)-2
        v = torch.floor(input_rescale * n + 0.5) / n
        v = (2 * v - 1)*input.max()  # [-1, 1]
        v = 0.5 * torch.log((1 + v) / (1 - v))  # arctanh
    else:
        input_rescale = (input + 1.0) / 2  # [0, 1]
        n = math.pow(2.0, bits) - 1
        v = torch.floor(input_rescale * n + 0.5) / n + 1
        v = 2 * v - 1  # [-1, 1]
        v = 0.5 * torch.log((1 + v) / (1 - v))  # arctanh
    return v


class LinearQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LinearQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            sf_new = self.bits - 1 - compute_integral_part(input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)


class LogQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LogQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            log_abs_input = torch.log(torch.abs(input))
            sf_new = self.bits - 1 - compute_integral_part(log_abs_input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = log_linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)


class NormalQuant(nn.Module):
    def __init__(self, name, bits, quant_func):
        super(NormalQuant, self).__init__()
        self.name = name
        self.bits = bits
        self.quant_func = quant_func

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        output = self.quant_func(input, self.bits)
        return output

    def __repr__(self):
        return '{}(bits={})'.format(self.__class__.__name__, self.bits)


def duplicate_model_with_quant(model, bits, overflow_rate=0.0, counter=10, type='linear'):
    """assume that original models has at least a nn.Sequential"""
    assert type in ['linear', 'minmax', 'log', 'tanh']
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():
            if isinstance(v, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.AvgPool2d)):
                l[k] = v
                if type == 'linear':
                    quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate,
                                              counter=counter)
                elif type == 'log':
                    # quant_layer = LogQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=log_minmax_quantize)
                elif type == 'minmax':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=min_max_quantize)
                else:
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=tanh_quantize)
                l['{}_{}_quant'.format(k, type)] = quant_layer
            else:
                l[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            model._modules[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        return model


## code from PIM analyzer
class Round_STE(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round_(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quant_tensor(tensor, bitwidth, interval, offset, asymmetric, quantized_offset=False):
    round_STE = Round_STE.apply
    if asymmetric:
        max_value = 2 ** bitwidth
        if quantized_offset:
            offset_int = round_STE(offset/interval).clamp(-max_value+1, 0) # according to source code
            tensor_int = round_STE(tensor/interval - offset_int).clamp(0, max_value-1)
            tensor_sim = (tensor_int+offset_int) * interval
        else:
            tensor_int = round_STE((tensor-offset)/interval).clamp(0, max_value-1)
            tensor_sim = tensor_int * interval + offset
    else:
        max_value = 2 ** (bitwidth - 1)
        tensor_int = round_STE(tensor/interval).clamp(-max_value, max_value-1)
        tensor_sim = tensor_int * interval
    return tensor_sim

def initialize_interval_offset(tensor, bitwidth, asymmetric, channelwise=False):
    """
    A simple min-max quantization.
    """
    if not channelwise:
        if asymmetric:
            interval = ((tensor.max()-tensor.min())/(2**(bitwidth)-1))
            offset = tensor.min()
        else:
            interval = (tensor.abs().max()/(2**(bitwidth-1)-1))
            offset = torch.tensor([0.])
    else:
        # WARNING: channelwise quantization should only be applied to weight tensor, 
        # since we assume its shape = OC, IC, kw, kh
        if asymmetric:
            w_max = torch.amax(tensor.data,[1,2,3],keepdim=True) # shape=OC,1,1,1
            w_min = torch.amin(tensor.data,[1,2,3],keepdim=True) # shape=OC,1,1,1
            interval = ((w_max-w_min)/(2**bitwidth-1)) # shape = OC,1,1,1
            offset = w_min # shape=OC,1,1,1
        else:
            w_max = torch.amax(tensor.data.abs(),[1,2,3],keepdim=True) # shape=OC,1,1,1
            interval = (w_max/(2**(bitwidth-1)-1))
            offset = torch.zeros_like(w_max)
    # Todo: move to quantized module
    # interval = nn.Parameter(interval)
    # offset = nn.Parameter(offset)
    return interval, offset

def dynamic_quant_tensor(tensor, bitwidth, asymmetric, quantized_offset=False):
    interval, offset = initialize_interval_offset(tensor, bitwidth, asymmetric)
    tensor_sim = quant_tensor(tensor, bitwidth, interval, offset, asymmetric, quantized_offset)
    return tensor_sim


class UnsignedQuantize(Function):
    """
    Unsigned Quantization.
    Scale is defined by |x|_max.
    Backward using STE. 
    """
    @staticmethod
    def forward(ctx, input, bit_width,interval):
        # ctx.save_for_backward(input)
        max_int=2**bit_width-1
        integer=torch.round_(input/interval).clamp_(0,max_int)
        out=integer*interval
        out.integer=integer
        out.interval=interval
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class SymmSignedQuantize(UnsignedQuantize):
    """
    Signed Quantization.
    Scale is defined by |x|_max.
    Backward using STE. 
    """
    @staticmethod
    def forward(ctx, input,bit_width,interval):
        # ctx.save_for_backward(input)
        max_int=2**(bit_width-1)-1
        integer=torch.round_(input/interval).clamp_(-max_int,max_int)
        out=integer*interval
        out.integer=integer
        out.interval=interval
        return out


class IntervalQuantizeIntO(Function):
    """
    Quantization with integer output and interval
    Backward using STE. 
    """
    @staticmethod
    def forward(ctx, input, interval, max_int, min_ini):
        integer=torch.round_(input/interval).clamp_(min_ini,max_int)
        out=integer
        out.interval=interval
        ctx.interval = interval
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output/ctx.interval, None, None, None

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
