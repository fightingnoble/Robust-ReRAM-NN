from typing import Dict, Any, Callable, AnyStr, List, Union,Tuple,Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .NN_accelerator import *
from .adc import _adc
from .dac import _quantize_dac

from torch.utils.hooks import RemovableHandle
from torch.jit import ScriptModule
# Some modules do the computation themselves using parameters
# or the parameters of children. Treat these as layers.
LAYER_MODULES = (torch.nn.MultiheadAttention,)
# These modules are not recorded during a forward pass. Handle them separately.
WRAPPER_MODULES = (nn.ParameterList, nn.ModuleList, ScriptModule)

# import torch.autograd.profiler as profiler

__all__ = [
    'crxb_Conv2d',
]
quantize_input = _quantize_dac.apply
quantize_weight = _quantize_dac.apply
adc = _adc.apply
print('!!! layer_q.py is imported\n')
minimal_num = 1e-12

from .quantization.layer_quantizer import SimpleCrxbQuantizer, RobustCrxbQuantizer
from .quantization.quant_functions import SymmSignedQuantize
from collections import OrderedDict

symmsignedquantize = SymmSignedQuantize.apply


class QuantizedLayer():
    mode = None
    # quantizer = None
    input_qbit = None
    x_scale = None
    weight_qbit = None
    weight_scale = None
    inter_activation_qbit = None
    y_scale = None
    bn_fused = False
    mapped = None

    # Interface
    def _qat_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        raise NotImplementedError

    def _static_quant_forward(self, inputs: Tensor, mapped_weight: Tensor, mapped_bias: Optional[Tensor]):
        raise NotImplemented
    
    def calibrate_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        raise NotImplementedError

    def from_float(cls,
                   mod,
                   layer_quantizer=RobustCrxbQuantizer,
                   crxb_cfg: Dict = {},
                   quantizer_kwargs: Dict = {}):
        raise NotImplemented

    def add_calib_hook(self, sequential=False, error_list:List=None):
        raise NotImplementedError

    # replace __repr__ with extra_repr
    def extra_repr(self) -> str:
        return super().extra_repr() + \
               "(I, W, A qbit:{} {} {})".format(
                   self.input_qbit if self.input_qbit < 32 else None,
                   self.weight_qbit if self.weight_qbit < 32 else None,
                   self.inter_activation_qbit if self.inter_activation_qbit < 32 else None)

    # Implementation
    def set_quant_parameters(self, input_qbit, weight_qbit, quantizer, inter_activation_qbit=None):
        self.inter_activation_qbit = inter_activation_qbit if inter_activation_qbit is not None else input_qbit
        self.weight_qbit = weight_qbit
        self.input_qbit = input_qbit
        self.quantizer = quantizer

    def reset_model(self):
        self.mapped = None

    def get_quant_weight_bias(self):
        # weight = module.weight if not module.mapped else module.weight_bk
        # bias = module.bias if not module.mapped else module.bias_bk
        # return self.quantizer.quant_weight_bias(weight, bias)
        return self.quantizer.quant_weight_bias(self.weight, self.bias)

    def forward(self, x):
        if self.mode == 'raw':
            out = super().forward(x)
        elif self.mode == "qat_forward":
            out = self._qat_forward(x, self.weight, self.bias)
        elif self.mode == "static_quant_forward":
            assert self.mapped, f"You should run map_weight before run _static_quant_forward for {self}"
            out = self._static_quant_forward(x, self.mapped_weight, self.mapped_bias)
        elif self.mode == "calibration_forward":
            out = self.calibrate_forward(x, self.weight, self.bias)
        else:
            raise NotImplementedError
        return out

    def convert_weight(self):
        # weight, bias = self.quantizer.quant_weight_bias(self.weight, self.bias)
        # self.weight_bk = self.weight.detach().clone()
        # self.weight.data.copy_(weight.data)
        # self.weight.interval = weight.interval
        # if self.bias is not None:
        #     self.bias_bk = self.bias.detach().clone()
        #     self.bias.data.copy_(bias.data)

        weight, bias = self.quantizer.quant_weight_bias(self.weight, self.bias)
        self.register_buffer('mapped_weight', weight)
        self.register_buffer('mapped_bias', bias)

        self.mapped = True
        self.mode = "static_quant_forward"


class QuantizeConv2d(QuantizedLayer, nn.Conv2d):

    def _qat_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        assert self.quantizer.calibrated, f"You should run calibrate_forward before run _qat_forward for {self}"
        weight_sim, bias_sim = self.quantizer.quant_weight_bias(weight, bias)
        x_sim = self.quantizer.quant_input(inputs)
        out_sim = F.conv2d(x_sim, weight_sim, bias_sim, self.stride,
                           self.padding, self.dilation, self.groups)
        out_sim = self.quantizer.quant_output(out_sim)
        return out_sim

    def _static_quant_forward(self, inputs: Tensor, mapped_weight: Tensor, mapped_bias: Optional[Tensor]):
        x_sim = self.quantizer.quant_input(inputs)
        out_sim = F.conv2d(x_sim, mapped_weight, mapped_bias, self.stride,
                           self.padding, self.dilation, self.groups)
        out_sim = self.quantizer.quant_output(out_sim)
        return out_sim

    def calibrate_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # move this into quantizer
        # assert self.weight_qbit and self.inter_activation_qbit, f"You should set the weight_qbit and bias_bits for {self}"
        op = lambda input, weight, bias: F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        # weight = module.weight if not module.mapped else module.weight_bk
        # bias = module.bias if not module.mapped else module.bias_bk
        # out_sim = self.quantizer.calibration(inputs, weight, bias, op)
        out_sim = self.quantizer.calibration(inputs, weight, bias, op)
        return out_sim

    def add_calib_hook(self, sequential=False, error_list:List=None):
        if sequential:
            def hook(module: crxb_Conv2d, inputs, output:List[torch.Tensor]):
                # weight = module.weight if not module.mapped else module.weight_bk
                # bias = module.bias if not module.mapped else module.bias_bk
                # ref_o = module._conv_forward(*inputs, weight, bias)
                ref_o = module._conv_forward(*inputs, self.weight, self.bias)
                # print("===={},{}====\n".format(type(inputs), type(output)))
                # assert False
                modified_o = output
                modified_o.ref_o = modified_o.new_tensor(ref_o)
                diff = F.mse_loss(output, ref_o, reduction="sum")
                if error_list is not None:
                    error_list.append([diff.item()])
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                else:
                    print("\nmax: {}\n".format((output - ref_o).abs().max()))
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                return modified_o
        else: 
            def hook(module: crxb_Conv2d, inputs, output:List[torch.Tensor]):
                # weight = module.weight if not module.mapped else module.weight_bk
                # bias = module.bias if not module.mapped else module.bias_bk
                if hasattr(inputs[0], 'ref_o'):
                    mod_in = [i.ref_o for i in inputs]
                else:
                    mod_in = inputs
                # ref_o = module._conv_forward(*mod_in, weight, bias)
                ref_o = module._conv_forward(*mod_in, self.weight, self.bias)
                modified_o = output
                modified_o.ref_o = modified_o.new_tensor(ref_o)
                diff = F.mse_loss(output, ref_o, reduction="sum")
                if error_list is not None:
                    error_list.append([diff.item()])
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                else:
                    print("\nmax: {}\n".format((output - ref_o).abs().max()))
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                return modified_o
        return self.register_forward_hook(hook)


class crxb_Conv2d(QuantizeConv2d):
    """
    This is the custom conv layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers
    Args:
        ir_drop(bool): switch that enables the ir drop calculation.
        device(torch.device): device index to select. It's a no-op if this argument is a negative integer or None.
        gmax(float): maximum conductance of the ReRAM.
        gmin(float): minimun conductance of the ReRAM.
        gwire(float): conductance of the metal wire.
        gload(float): load conductance of the ADC and DAC.
        vdd(float): supply voltage.
        scaler_dw(float): weight quantization scaler to reduce the influence of the ir drop.
        enable_noise(bool): switch to enable stochastic_noise.
        freq(float): operating frequency of the ReRAM crossbar.
        temp(float): operating temperature of ReRAM crossbar.
        crxb_size(int): size of the crossbar.
        enable_SAF(bool): switch to enable SAF
        enable_ec_SAF(bool): switch to enable SAF error correction.
        input_qbit(int): quantization resolution of the crossbar.
        weight_qbit(int): quantization resolution of the crossbar.
        inter_activation_qbit(int): quantization resolution of the crossbar.
        is_first_layer(bool):
        is_last_layer(bool):
        noise_amp(float):
        noise_mean(float):
        noise_var(float):
        random_noise(bool):
    """

    layer_type = "any"
    q_type = "any"
    array_type = "NN_array"

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 q_type='uniform',
                 input_qbit=32,
                 weight_qbit=32,
                 inter_activation_qbit=32,
                 is_first_layer=False,
                 is_last_layer=False,
                 **crxb_cfg):

        super(crxb_Conv2d, self).__init__(in_channels, out_channels, 
                                          kernel_size, stride, padding, dilation, groups, bias)
        # =================== Quantizer defination ===================
        # create new object element for layer notation: add: 21-06-12
        self.q_type = q_type
        if self.q_type in ['robust', 'robust_batch']:
            self.layer_type = 'learnable'
        else:
            self.layer_type = 'fixed'

        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        self.input_qbit = input_qbit if not (self.is_first_layer
                                             and input_qbit < 8) else 8
        self.weight_qbit = weight_qbit
        self.inter_activation_qbit = inter_activation_qbit if not (
            self.is_last_layer and inter_activation_qbit < 8) else 8
        self.calibration = False

        self.quantizer = SimpleCrxbQuantizer(self.weight_qbit, self.input_qbit,
                                             8, self.inter_activation_qbit)

        assert self.groups == 1, "currently not support grouped convolution for custom conv"

        ################## Crossbar conversion #############################
        self.crxb_size = crxb_cfg['crxb_size']
        # 210626 modifidation: data copy optimization
        weight_flatten_rows = self.in_channels * torch.cumprod(
            torch.tensor(self.kernel_size), 0)[-1].item()
        weight_flatten_cols = self.out_channels
        self.crxb_row, self.crxb_row_pads = num_pad(weight_flatten_rows,
                                                    self.crxb_size)
        self.crxb_col, self.crxb_col_pads = num_pad(weight_flatten_cols,
                                                    self.crxb_size)
        # p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        self.w_pad = [0, self.crxb_row_pads, 0, self.crxb_col_pads]
        self.input_pad = [0, 0, 0, self.crxb_row_pads]

        ################# Hardware conversion ##############################
        # weight and input levels
        # ReRAM cells
        # moved to NN_array
        self.phy = NNarray(crxb_row=self.crxb_row,
                           crxb_col=self.crxb_col,
                           output_select=torch.arange(self.out_channels),
                           bias=bias,
                           n_lvl_v=2**input_qbit,
                           **crxb_cfg)

    def reset_model(self):
        super().reset_model()
        self.quantizer.reset()

    def op1(self, input_quan, weight_quan):
        # with profiler.record_function("computation"):
        # 2. Perform the computation between input voltage and weight conductance

        # 20210321: modify: add mapping function to adapt different mapping methods in the future.
        # 2.1 flatten and unfold the weight and input
        # 2.2. add paddings
        # 2.3. reshape to crxb size
        input_crxb, weight_crxb = self.map(input_quan, weight_quan)

        # 2.4. compute matrix multiplication followed by reshapes
        # (N, C, crxb_row, L)
        output_crxb = self.phy(input_crxb, weight_crxb)
        return output_crxb

    def map(self, inputs=None, weight_i=None):
        return_list = []

        # 2.1 flatten and unfold the weight and input
        input_unfold = F.unfold(inputs,
                                kernel_size=self.kernel_size,
                                dilation=self.dilation,
                                padding=self.padding,
                                stride=self.stride)

        # 2.2. add paddings
        input_padded = F.pad(input_unfold,
                             self.input_pad,
                             mode='constant',
                             value=0)
        # 2.3. reshape to crxb size
        input_crxb = input_padded.view(inputs.shape[0], 1, self.crxb_row,
                                       self.crxb_size, input_padded.shape[2])
        return_list.append(input_crxb)

        # 2.1 flatten and unfold the weight and input
        weight_flatten = weight_i.view(self.out_channels, -1)

        # 2.2. add paddings
        weight_padded = F.pad(weight_flatten,
                              self.w_pad,
                              mode='constant',
                              value=0)
        # 2.3. reshape to crxb size
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row,
                                         self.crxb_size).transpose(1, 2)
        return_list.append(weight_crxb)
        return return_list

    def demap(self, weight_crxb):
        # 2.3. reshape to crxb size
        weight_padded = weight_crxb.transpose(1, 2).view(
            self.crxb_col * self.crxb_size, self.crxb_row * self.crxb_size)
        # 2.2. add paddings
        weight_flatten = weight_padded[self.w_pad[2]:-self.w_pad[3],
                                       self.w_pad[0]:-self.w_pad[1]]
        # 2.1 flatten and unfold the weight and input
        weight_i = weight_flatten.view(self.out_channels, self.in_channels,
                                       *self.kernel_size)
        return weight_i

    # interface implementation
    def _qat_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        assert self.quantizer.calibrated, f"You should run calibrate_forward before run _qat_forward for {self}"

        # with profiler.record_function("input data and weight quantization"):
        # 1. input data and weight quantization
        w_sim, bias_sim = self.quantizer.quant_weight_bias(
            weight, bias)
        x_sim = self.quantizer.quant_input(inputs)

        weight_quan = w_sim
        input_quan = x_sim

        output_crxb = self.op1(input_quan, weight_quan)

        # True process
        # TIA_vout = output current * amplifier
        # amplifier = scaler / self.phy.delta_g * (self.n_lvl_v - 1)
        # DAC_vout = DAC(TIA_vout)
        # out = DAC_out / sef.phy.Vdd

        # Simplified process
        # with profiler.record_function("perform ADC operation"):
        # 3. perform ADC operation (i.e., current to digital conversion)
        quant_i = self.quantizer.quant_output(output_crxb)

        # with profiler.record_function("get output with partial sum and bias and rescale"):
        # (N, C, L)
        output_sum = self.phy.partial_acc(quant_i)

        delta_x = x_sim.interval
        delta_w = w_sim.interval
        delta_y = delta_w * delta_x / (self.phy.delta_v * self.phy.delta_g)
        output = output_sum * delta_y

        if self.bias is not None:
            # (N, C, L)
            output += bias_sim.unsqueeze(1)

        h_out = int((inputs.shape[2] - self.kernel_size[0] +
                     2 * self.padding[0]) / self.stride[0] + 1)
        w_out = int((inputs.shape[3] - self.kernel_size[0] +
                     2 * self.padding[0]) / self.stride[0] + 1)

        # (N, C, H, W)
        output = output.view(output.shape[0:2] + (h_out, w_out))
        return output

    def _static_quant_forward(self, inputs: Tensor, mapped_weight: Tensor, mapped_bias: Optional[Tensor]):
        # with profiler.record_function("input data and weight quantization"):
        # 1. input data and weight quantization
        # print("simple!!!")
        w_sim, bias_sim = mapped_weight, mapped_bias
        x_sim = self.quantizer.quant_input(inputs)

        weight_quan = w_sim
        input_quan = x_sim

        output_crxb = self.op1(input_quan, weight_quan)

        # True process
        # TIA_vout = output current * amplifier
        # amplifier = scaler / self.phy.delta_g * (self.n_lvl_v - 1)
        # DAC_vout = DAC(TIA_vout)
        # out = DAC_out / sef.phy.Vdd

        # Simplified process
        # with profiler.record_function("perform ADC operation"):
        # 3. perform ADC operation (i.e., current to digital conversion)
        quant_i = self.quantizer.quant_output(output_crxb)

        # with profiler.record_function("get output with partial sum and bias and rescale"):
        # (N, C, L)
        output_sum = self.phy.partial_acc(quant_i)

        delta_x = x_sim.interval
        delta_w = w_sim.interval
        delta_y = delta_w * delta_x / (self.phy.delta_v * self.phy.delta_g)
        output = output_sum * delta_y

        if self.bias is not None:
            # (N, C, L)
            output += bias_sim.unsqueeze(1)

        h_out = int((inputs.shape[2] - self.kernel_size[0] +
                     2 * self.padding[0]) / self.stride[0] + 1)
        w_out = int((inputs.shape[3] - self.kernel_size[0] +
                     2 * self.padding[0]) / self.stride[0] + 1)

        # (N, C, H, W)
        output = output.view(output.shape[0:2] + (h_out, w_out))
        return output

    def calibrate_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # weight = module.weight if not module.mapped else module.weight_bk
        # bias = module.bias if not module.mapped else module.bias_bk

        quant_i, x_sim, w_sim = self.quantizer.calibration(
            inputs, weight, bias, self.op1)
        # with profiler.record_function("get output with partial sum and bias and rescale"):
        # (N, C, L)
        output_sum = self.phy.partial_acc(quant_i)

        delta_x = x_sim.interval
        delta_w = w_sim.interval
        delta_y = delta_w * delta_x / (self.phy.delta_v * self.phy.delta_g)
        output = output_sum * delta_y

        if self.bias is not None:
            # (N, C, L)
            bias_sim = self.quantizer.quant_bias(self.bias)
            output += bias_sim.unsqueeze(1)

        h_out = int((inputs.shape[2] - self.kernel_size[0] +
                     2 * self.padding[0]) / self.stride[0] + 1)
        w_out = int((inputs.shape[3] - self.kernel_size[0] +
                     2 * self.padding[0]) / self.stride[0] + 1)

        # (N, C, H, W)
        output = output.view(output.shape[0:2] + (h_out, w_out))
        return output

    def add_calib_hook(self, sequential=False, error_list:List=None):
        if sequential:
            def hook(module: crxb_Conv2d, inputs, output:List[torch.Tensor]):
                # weight = module.weight if not module.mapped else module.weight_bk
                # bias = module.bias if not module.mapped else module.bias_bk
                # ref_o = module._conv_forward(*inputs, weight, bias)
                ref_o = module._conv_forward(*inputs, self.weight, self.bias)
                self.phy.mode = "clean"
                quant_ref_o = module.forward(*inputs)
                self.phy.mode = "noisy"
                # print("===={},{}====\n".format(type(inputs), type(output)))
                # assert False
                modified_o = output 
                modified_o.quant_ref_o = modified_o.new_tensor(quant_ref_o)
                modified_o.ref_o = modified_o.new_tensor(ref_o)
                diff = F.mse_loss(output, ref_o, reduction="sum")
                diff1 = F.mse_loss(quant_ref_o, ref_o, reduction="sum")
                diff2 = F.mse_loss(quant_ref_o, output, reduction="sum")
                if error_list is not None:
                    error_list.append([diff.item(), diff1.item(), diff2.item()])
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                else:
                    print("\nmax: {}\n".format((output - ref_o).abs().max()))
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                return modified_o
        else: 
            def hook(module: crxb_Conv2d, inputs, output:List[torch.Tensor]):
                # weight = module.weight if not module.mapped else module.weight_bk
                # bias = module.bias if not module.mapped else module.bias_bk
                if hasattr(inputs[0], 'ref_o'):
                    mod_in = [i.ref_o for i in inputs]
                else:
                    mod_in = inputs
                # ref_o = module._conv_forward(*mod_in, weight, bias)
                ref_o = module._conv_forward(*mod_in, self.weight, self.bias)
                self.phy.mode = "clean"
                quant_ref_o = module.forward(*mod_in)
                self.phy.mode = "noisy"
                modified_o = output 
                modified_o.quant_ref_o = modified_o.new_tensor(quant_ref_o)
                modified_o.ref_o = modified_o.new_tensor(ref_o)
                diff = F.mse_loss(output, ref_o, reduction="sum")
                diff1 = F.mse_loss(quant_ref_o, ref_o, reduction="sum")
                diff2 = F.mse_loss(quant_ref_o, output, reduction="sum")
                if error_list is not None:
                    error_list.append([diff.item(), diff1.item(), diff2.item()])
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                else:
                    print("\nmax: {}\n".format((output - ref_o).abs().max()))
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                return modified_o
        return self.register_forward_hook(hook)

    @classmethod
    def from_float(cls,
                   mod,
                   layer_quantizer=RobustCrxbQuantizer,
                   crxb_cfg: Dict = {},
                   quantizer_kwargs: Dict = {}):
        cfg = {
            "in_channels": mod.in_channels,
            "out_channels": mod.out_channels,
            "kernel_size": mod.kernel_size,
            "stride": mod.stride,
            "padding": mod.padding,
            "dilation": mod.dilation,
            "transposed": mod.transposed,
            "output_padding": mod.output_padding,
            "groups": mod.groups,
            "bias": mod.bias is not None,
            "padding_mode": mod.padding_mode,
        }
        # print(crxb_cfg)
        if not crxb_cfg:
            crxb_cfg = copy.deepcopy(DEFAULT_CRSB_CFGS)
        crxb_layer = cls(**cfg, **crxb_cfg)
        crxb_layer.weight.data.copy_(mod.weight.data)
        crxb_layer.bias.data.copy_(mod.bias.data)
        if layer_quantizer is RobustCrxbQuantizer:
            if not quantizer_kwargs.get("gamma"):
                quantizer_kwargs.update({"gamma": 0.})
        crxb_layer.quantizer = layer_quantizer(crxb_layer.weight_qbit,
                                              crxb_layer.input_qbit, 8,
                                              **quantizer_kwargs)
        crxb_layer.mode = "raw"
        return crxb_layer


class QuantizeLinear(QuantizedLayer, nn.Linear):

    def _qat_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        assert self.quantizer.calibrated, f"You should run calibrate_forward before run _qat_forward for {self}"
        weight_sim, bias_sim = self.quantizer.quant_weight_bias(
            weight, bias)
        x_sim = self.quantizer.quant_input(inputs)
        out_sim = F.linear(x_sim, weight_sim, bias_sim)
        out_sim = self.quantizer.quant_output(out_sim)
        return out_sim

    def calibrate_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # move this into quantizer
        # assert self.weight_qbit is not None and self.inter_activation_qbit is not None, f"You should set the weight_qbit and bias_bits for {self}"
        op = lambda input, weight, bias: F.linear(input, weight, bias)
        out_sim = self.quantizer.calibration(inputs, weight, bias, op)
        return out_sim

    def _static_quant_forward(self, inputs: Tensor, mapped_weight: Tensor, mapped_bias: Optional[Tensor]):
        x_sim = self.quantizer.quant_input(inputs)
        out_sim = F.linear(x_sim, mapped_weight, mapped_bias)
        out_sim = self.quantizer.quant_output(out_sim)
        return out_sim

    def add_calib_hook(self, sequential=False, error_list:List=None):
        if sequential:
            def hook(module: crxb_Conv2d, inputs, output:List[torch.Tensor]):
                # weight = module.weight if not module.mapped else module.weight_bk
                # bias = module.bias if not module.mapped else module.bias_bk
                # ref_o = F.linear(*inputs, weight, bias)
                ref_o = F.linear(*inputs, self.weight, self.bias)
                # print("===={},{}====\n".format(type(inputs), type(output)))
                # assert False
                modified_o = output 
                modified_o.ref_o = modified_o.new_tensor(ref_o)
                diff = F.mse_loss(output, ref_o, reduction="sum")
                if error_list is not None:
                    error_list.append([diff.item()])
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                else:
                    print("\nmax: {}\n".format((output - ref_o).abs().max()))
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                return modified_o
        else: 
            def hook(module: crxb_Conv2d, inputs, output:List[torch.Tensor]):
                # weight = module.weight if not module.mapped else module.weight_bk
                # bias = module.bias if not module.mapped else module.bias_bk
                if hasattr(inputs[0], 'ref_o'):
                    mod_in = [i.ref_o for i in inputs]
                else:
                    mod_in = inputs
                # ref_o = F.linear(*mod_in, weight, bias)
                ref_o = F.linear(*mod_in, self.weight, self.bias)
                modified_o = output
                modified_o.ref_o = modified_o.new_tensor(ref_o)
                diff = F.mse_loss(output, ref_o, reduction="sum")
                if error_list is not None:
                    error_list.append([diff.item()])
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                else:
                    print("\nmax: {}\n".format((output - ref_o).abs().max()))
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                return modified_o
        return self.register_forward_hook(hook)


class crxb_Linear(QuantizeLinear):
    """
    This is the custom linear layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers
    Args:
        ir_drop(bool): switch that enables the ir drop calculation.
        device(torch.device): device index to select. It's a no-op if this argument is a negative integer or None.
        gmax(float): maximum conductance of the ReRAM.
        gmin(float): minimun conductance of the ReRAM.
        gwire(float): conductance of the metal wire.
        gload(float): load conductance of the ADC and DAC.
        vdd(float): supply voltage.
        scaler_dw(float): weight quantization scaler to reduce the influence of the ir drop.
        enable_noise(bool): switch to enable stochastic_noise.
        freq(float): operating frequency of the ReRAM crossbar.
        temp(float): operating temperature of ReRAM crossbar.
        crxb_size(int): size of the crossbar.
        enable_SAF(bool): switch to enable SAF
        enable_ec_SAF(bool): switch to enable SAF error correction.
        input_qbit(int): quantization resolution of the crossbar.
        weight_qbit(int): quantization resolution of the crossbar.
        inter_activation_qbit(int): quantization resolution of the crossbar.
        is_first_layer(bool):
        is_last_layer(bool):
        noise_amp(float):
        noise_mean(float):
        noise_var(float):
        random_noise(bool):
    """

    layer_type = "any"
    q_type = "any"
    array_type = "NN_array"

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 q_type='uniform',
                 input_qbit=32,
                 weight_qbit=32,
                 inter_activation_qbit=32,
                 is_first_layer=False,
                 is_last_layer=False,
                 **crxb_cfg):
        super(crxb_Linear, self).__init__(in_features, out_features, bias)
        # =================== Quantizer defination ===================
        # create new object element for layer notation: add: 21-06-12
        self.q_type = q_type
        if self.q_type in ['robust', 'robust_batch']:
            self.layer_type = 'learnable'
        else:
            self.layer_type = 'fixed'

        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        self.input_qbit = input_qbit if not (self.is_first_layer
                                             and input_qbit < 8) else 8
        self.weight_qbit = weight_qbit
        self.inter_activation_qbit = inter_activation_qbit if not (
            self.is_last_layer and inter_activation_qbit < 8) else 8
        self.calibration = False

        self.quantizer = SimpleCrxbQuantizer(self.weight_qbit, self.input_qbit,
                                             8, self.inter_activation_qbit)

        ################## Crossbar conversion #############################
        self.crxb_size = crxb_cfg['crxb_size']

        self.crxb_row, self.crxb_row_pads = num_pad(self.in_features,
                                                    self.crxb_size)
        self.crxb_col, self.crxb_col_pads = num_pad(self.out_features,
                                                    self.crxb_size)
        # p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3) for last 1, 2, 3 dimension
        self.w_pad = [0, self.crxb_row_pads, 0, self.crxb_col_pads]
        self.input_pad = [0, self.crxb_row_pads]

        ################# Hardware conversion ##############################
        # weight and input levels
        # ReRAM cells
        # moved to NN_array
        self.phy = NNarray(crxb_row=self.crxb_row,
                           crxb_col=self.crxb_col,
                           output_select=torch.arange(self.out_features),
                           bias=bias,
                           input_qbit=input_qbit,
                           weight_qbit=weight_qbit,
                           inter_activation_qbit=inter_activation_qbit,
                           scaler_dy=1,
                           is_first_layer=is_first_layer,
                           is_last_layer=is_last_layer,
                           **crxb_cfg)

    def reset_model(self):
        super().reset_model()
        self.quantizer.reset()

    def op1(self, input_quan, weight_quan):
        # 2. Perform the computation between input voltage and weight conductance
        # 2.1. skip the input unfold and weight flatten for fully-connected layers
        # 2.2. add paddings
        # 2.3. reshape
        input_crxb, weight_crxb = self.map(input_quan, weight_quan)

        # 2.4. compute matrix multiplication followed by reshapes
        # (N, C, crxb_row)
        output_crxb = self.phy(input_crxb,weight_crxb).squeeze(-1)
        return output_crxb

    def map(self, inputs=None, weight_i=None):
        return_list = []
        # 2. Perform the computation between input voltage and weight conductance
        # 2.1. skip the input unfold and weight flatten for fully-connected layers

        # 2.2. add paddings
        input_padded = F.pad(inputs, self.input_pad, mode='constant', value=0)
        # 2.3. reshape
        input_crxb = input_padded.view(inputs.shape[0], 1, self.crxb_row,
                                       self.crxb_size, 1)
        return_list.append(input_crxb)

        # 2.2. add paddings
        weight_padded = F.pad(weight_i, self.w_pad, mode='constant', value=0)
        # 2.3. reshape
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row,
                                         self.crxb_size).transpose(1, 2)
        return_list.append(weight_crxb)
        return return_list

    def demap(self, weight_crxb):
        # 2.3. reshape to crxb size
        weight_padded = weight_crxb.transpose(1, 2).view(
            self.crxb_col * self.crxb_size, self.crxb_row * self.crxb_size)
        # 2.2. add paddings
        weight_i = weight_padded[self.w_pad[2]:-self.w_pad[3],
                                 self.w_pad[0]:-self.w_pad[1]]
        return weight_i

    # interface implementation
    def _qat_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        assert self.quantizer.calibrated, f"You should run calibrate_forward before run _qat_forward for {self}"

        # 1. input data and weight quantization
        w_sim, bias_sim = self.quantizer.quant_weight_bias(
            weight, bias)
        x_sim = self.quantizer.quant_input(inputs)

        weight_quan = w_sim
        input_quan = x_sim

        output_crxb = self.op1(input_quan, weight_quan)

        # True process
        # TIA_vout = output current * amplifier
        # amplifier = scaler / self.phy.delta_g * (self.n_lvl_v - 1)
        # DAC_vout = DAC(TIA_vout)
        # out = DAC_out / sef.phy.Vdd

        # Simplified process
        # 3. perform ADC operation (i.e., current to digital conversion)
        quant_i = self.quantizer.quant_output(output_crxb)

        # (N, C)
        output_sum = self.phy.partial_acc(quant_i)

        delta_x = x_sim.interval
        delta_w = w_sim.interval
        delta_y = delta_w * delta_x / (self.phy.delta_v * self.phy.delta_g)
        output = output_sum * delta_y

        if self.bias is not None:
            output += bias_sim
        return output

    def _static_quant_forward(self, inputs: Tensor, mapped_weight: Tensor, mapped_bias: Optional[Tensor]):
        # 1. input data and weight quantization
        w_sim, bias_sim = mapped_weight, mapped_bias
        x_sim = self.quantizer.quant_input(inputs)

        weight_quan = w_sim
        input_quan = x_sim

        output_crxb = self.op1(input_quan, weight_quan)

        # True process
        # TIA_vout = output current * amplifier
        # amplifier = scaler / self.phy.delta_g * (self.n_lvl_v - 1)
        # DAC_vout = DAC(TIA_vout)
        # out = DAC_out / sef.phy.Vdd

        # Simplified process
        # 3. perform ADC operation (i.e., current to digital conversion)
        quant_i = self.quantizer.quant_output(output_crxb)

        # (N, C)
        output_sum = self.phy.partial_acc(quant_i)

        delta_x = x_sim.interval
        delta_w = w_sim.interval
        delta_y = delta_w * delta_x / (self.phy.delta_v * self.phy.delta_g)
        output = output_sum * delta_y

        if self.bias is not None:
            output += bias_sim
        return output

    def calibrate_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        quant_i, x_sim, w_sim = self.quantizer.calibration(
            inputs, weight, bias, self.op1)
        # with profiler.record_function("get output with partial sum and bias and rescale"):
        # (N, C, L)
        output_sum = self.phy.partial_acc(quant_i)

        delta_x = x_sim.interval
        delta_w = w_sim.interval
        delta_y = delta_w * delta_x / (self.phy.delta_v * self.phy.delta_g)
        output = output_sum * delta_y

        if self.bias is not None:
            # (N, C, L)
            bias_sim = self.quantizer.quant_bias(self.bias)
            output += bias_sim

        return output

    def _clean_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        assert self.quantizer.calibrated, f"You should run calibrate_forward before run _qat_forward for {self}"

        # 1. input data and weight quantization
        w_sim, bias_sim = self.quantizer.quant_weight_bias(
            weight, bias)
        x_sim = self.quantizer.quant_input(inputs)

        weight_quan = w_sim
        input_quan = x_sim

        output_crxb = self.op1(input_quan, weight_quan)

        # True process
        # TIA_vout = output current * amplifier
        # amplifier = scaler / self.phy.delta_g * (self.n_lvl_v - 1)
        # DAC_vout = DAC(TIA_vout)
        # out = DAC_out / sef.phy.Vdd

        # Simplified process
        # 3. perform ADC operation (i.e., current to digital conversion)
        quant_i = self.quantizer.quant_output(output_crxb)

        # (N, C)
        output_sum = self.phy.partial_acc(quant_i)

        delta_x = x_sim.interval
        delta_w = w_sim.interval
        delta_y = delta_w * delta_x / (self.phy.delta_v * self.phy.delta_g)
        output = output_sum * delta_y

        if self.bias is not None:
            output += bias_sim
        return output

    def add_calib_hook(self, sequential=False, error_list:List=None):
        if sequential:
            def hook(module: crxb_Conv2d, inputs, output:List[torch.Tensor]):
                # weight = module.weight if not module.mapped else module.weight_bk
                # bias = module.bias if not module.mapped else module.bias_bk
                # ref_o = F.linear(*inputs, weight, bias)
                ref_o = F.linear(*inputs, self.weight, self.bias)
                self.phy.mode = "clean"
                quant_ref_o = module.forward(*inputs)
                self.phy.mode = "noisy"
                # print("===={},{}====\n".format(type(inputs), type(output)))
                # assert False
                modified_o = output 
                modified_o.quant_ref_o = modified_o.new_tensor(quant_ref_o)
                modified_o.ref_o = modified_o.new_tensor(ref_o)
                diff = F.mse_loss(output, ref_o, reduction="sum")
                diff1 = F.mse_loss(quant_ref_o, ref_o, reduction="sum")
                diff2 = F.mse_loss(quant_ref_o, output, reduction="sum")
                if error_list is not None:
                    error_list.append([diff.item(), diff1.item(), diff2.item()])
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                else:
                    print("\nmax: {}\n".format((output - ref_o).abs().max()))
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                return modified_o
        else: 
            def hook(module: crxb_Conv2d, inputs, output:List[torch.Tensor]):
                # weight = module.weight if not module.mapped else module.weight_bk
                # bias = module.bias if not module.mapped else module.bias_bk
                if hasattr(inputs[0], 'ref_o'):
                    mod_in = [i.ref_o for i in inputs]
                else:
                    mod_in = inputs
                # ref_o = F.linear(*mod_in, weight, bias)
                ref_o = F.linear(*mod_in, self.weight, self.bias)
                self.phy.mode = "clean"
                quant_ref_o = module.forward(*mod_in)
                self.phy.mode = "noisy"
                modified_o = output 
                modified_o.quant_ref_o = modified_o.new_tensor(quant_ref_o)
                modified_o.ref_o = modified_o.new_tensor(ref_o)
                diff = F.mse_loss(output, ref_o, reduction="sum")
                diff1 = F.mse_loss(quant_ref_o, ref_o, reduction="sum")
                diff2 = F.mse_loss(quant_ref_o, output, reduction="sum")
                if error_list is not None:
                    error_list.append([diff.item(), diff1.item(), diff2.item()])
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                else:
                    print("\nmax: {}\n".format((output - ref_o).abs().max()))
                    print("\n{}: total: {}\n".format(error_list[0], diff))
                return modified_o
        return self.register_forward_hook(hook)

    @classmethod
    def from_float(cls,
                   mod,
                   layer_quantizer=RobustCrxbQuantizer,
                   crxb_cfg: Dict = {},
                   quantizer_kwargs: Dict = {}):
        cfg = {
            "in_features": mod.in_features,
            "out_features": mod.out_features,
            "bias": mod.bias is not None,
        }
        # print(crxb_cfg)
        if not crxb_cfg:
            crxb_cfg = copy.deepcopy(DEFAULT_CRSB_CFGS)
        crxb_layer = cls(**cfg, **crxb_cfg)
        crxb_layer.weight.data.copy_(mod.weight.data)
        crxb_layer.bias.data.copy_(mod.bias.data)
        if layer_quantizer is RobustCrxbQuantizer:
            if not quantizer_kwargs.get("gamma"):
                quantizer_kwargs.update({"gamma": 0.})
        crxb_layer.quantizer = layer_quantizer(crxb_layer.weight_qbit,
                                              crxb_layer.input_qbit, 8,
                                              **quantizer_kwargs)
        crxb_layer.mode = "raw"
        return crxb_layer


DEFAULT_CRSB_CFGS = {
    'gmax': 0.000333,
    'gmin': 3.33e-07,
    'gwire': 0.0357,
    'gload': 0.25,
    'q_type': 'robust_batch',
    'input_qbit': 8,
    'weight_qbit': 8,
    'inter_activation_qbit': 8,
    'vdd': 3.3,
    'freq': 10000000.0,
    'temp': 300,
    'crxb_size': 64,
    'enable_SAF': False,
    'enable_ec_SAF': False,
    'enable_noise': False,
    'noise_amp': 0.12,
    'random_noise': True,
    'ir_drop': False,
    'device': 'cuda',
    'enable_ec_Var': False,
    'p_fix': 0.1
}

DEFAULT_STATIC_QUANT_MODULE_MAPPINGS: Dict[Callable, Any] = {
    nn.Linear: crxb_Linear,
    nn.Conv2d: crxb_Conv2d
}


def convert(module: nn.Module,
            mapping=None,
            inplace=False,
            layer_quantizer=RobustCrxbQuantizer,
            wrap_fc=False,
            prefix: str = '',
            wrapped_modules=None,
            quantizer_kwargs=None,
            crxb_cfg=None):
    r"""Converts submodules in input module to a different module according to `mapping`
    by calling `from_float` method on the target module class

    Args:
        module: input module
        mapping: a dictionary that maps from source module type to target
                 module type, can be overwritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated

    """
    # print(crxb_cfg)
    # print("Before wrapped\n"+"*"*10+"\n",module)
    if mapping is None:
        mapping = copy.deepcopy(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)
    # module.named_modules
    # module.named_children
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    if wrapped_modules is None:
        wrapped_modules = OrderedDict()
    if quantizer_kwargs is None:
        quantizer_kwargs = {}
    for name, mod in module.named_children():
        # both fused modules and observed custom modules are
        # swapped as one unit
        # print(type(mod))
        submodule_prefix = prefix + ('.' if prefix else '') + name
        if type(mod) in mapping:
            new_mod = mapping[type(mod)].from_float(
                mod,
                layer_quantizer=layer_quantizer,
                crxb_cfg=crxb_cfg,
                quantizer_kwargs=quantizer_kwargs)
            reassign[name] = new_mod
            wrapped_modules[submodule_prefix] = new_mod
        convert(
            mod,
            mapping,
            True,  # inplace
            layer_quantizer,
            wrap_fc,
            submodule_prefix,
            wrapped_modules,
            quantizer_kwargs,
            crxb_cfg)
    for key, value in reassign.items():
        module._modules[key] = value
    return module


def convert_weight(model:nn.Module, wrapped_modules: Dict[AnyStr, QuantizedLayer], inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    for name,module in wrapped_modules.items():
        module.apply(m_convert_weight)


def reset_model(model:nn.Module, wrapped_modules: Dict[AnyStr, QuantizedLayer]):
    for name,module in wrapped_modules.items():
        module.apply(m_reset_model)


def m_convert_weight(module:nn.Module):
    if isinstance(module, QuantizedLayer):
        module.convert_weight()


def m_reset_model(module: nn.Module):
    if isinstance(module, QuantizedLayer):
        module.reset_model()

import torchinfo
def apply_hooks(
    named_module: Tuple[str, nn.Module],
    orig_model: nn.Module,
    batch_dim: Optional[int],
    summary_list: List[Any],
    idx: Dict[int, int],
    hooks: Optional[List[RemovableHandle]],
    curr_depth: int = 0,
) -> None:
    """
    If input_data is provided, recursively adds hooks to all layers of the model.
    Else, fills summary_list with layer info without computing a
    forward pass through the network.
    """
    # Fallback is used if the layer's pre-hook is never called, for example in
    # ModuleLists or Sequentials.
    var_name, module = named_module

    def pre_hook(module: nn.Module, inputs: Any) -> None:
        """Create a LayerInfo object to aggregate information about that layer."""
        del inputs
        idx[curr_depth] = idx.get(curr_depth, 0) + 1
        # add code here
        summary_list.append()

    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Update LayerInfo after forward pass."""
        del module
        # add code here

    submodules = [m for m in module.modules() if m is not orig_model]
    if module != orig_model or isinstance(module, LAYER_MODULES) or not submodules:
        if hooks is None or isinstance(module, WRAPPER_MODULES):
            pre_hook(module, None)
        else:
            hooks.append(module.register_forward_pre_hook(pre_hook))
            hooks.append(module.register_forward_hook(hook))

    for child in module.named_children():
        apply_hooks(
            child, orig_model, batch_dim, summary_list, idx, hooks, curr_depth + 1
        )


# def model_conversion(model_ref: nn.Module, model_out: nn.Module,
#                      calibration_inputs):
#     model_dict = model_out.state_dict()
#     pretrained_dict = model_ref.state_dict()
#     for k, v in model_dict.items():
#         if k in pretrained_dict:
#             model_dict[k].copy_(pretrained_dict[k].data)
#     model_out.mode = "calibration_forward"
#     model_out(*calibration_inputs)
#     model_out.mode = "qat_forward"


if __name__ == "__main__":
    cfg_conv = (3, 32, 3)
    cfg_linear = (26*26*32)
    class conv_or_linear(nn.Module):
        def __init__(self):
            super(conv_or_linear, self).__init__()
            self.conv = nn.Conv2d(*cfg_conv)
        
        def forward(self, x):
            x = self.conv(x)
            return x
    model = conv_or_linear()
    feature = torch.rand(1, 3, 28, 28)
    crxb_cfg = copy.deepcopy(DEFAULT_CRSB_CFGS)
    crxb_cfg['weight_qbit'] = 8
    wrapped_modules = OrderedDict()
    model_crxb = convert(
        model,
        inplace=False,
        layer_quantizer=
        RobustCrxbQuantizer, # RobustCrxbQuantizer, SimpleCrxbQuantizer
        wrap_fc=False,
        wrapped_modules=wrapped_modules,
        crxb_cfg=crxb_cfg)

    model_crxb.cuda()
    feature = feature.cuda()
    model.cuda()

    for name, module in wrapped_modules.items():
        module.mode = 'calibration_forward'
    model_crxb(feature)
    for name, module in model_crxb.named_modules():
        if name in wrapped_modules.keys():
            print(f"{name}: {module.quantizer}")
            module.mode = 'qat_forward'
    print("calibration finished")

    hooks = [] 
    for name,module in wrapped_modules.items():
        hook = module.add_calib_hook(True)
        hooks.append(hook)

    q_o = model_crxb(feature)
    for hook in hooks:
        hook.remove()
    ref_o = model(feature)
    print((q_o - ref_o).abs().max())
    diff = F.mse_loss(q_o, ref_o, reduction="sum")
    print(diff)
