import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .NN_accelerator import *
from .adc import _adc
from .dac import _quantize_dac

# import torch.autograd.profiler as profiler

__all__ = ['crxb_Conv2d', ]
quantize_input = _quantize_dac.apply
quantize_weight = _quantize_dac.apply
adc = _adc.apply
print('!!! layer_q.py is imported\n')
minimal_num = 1e-12

from .quantization.layer_quantizer import SimpleCrxbQuantizer
from .quantization.quant_functions import SymmSignedQuantize

symmsignedquantize = SymmSignedQuantize.apply


class UniformQ():
    def __init__(self, bitwidth=8) -> None:
        self.bitwidth = bitwidth

    def __call__(self, x) -> Tensor:
        interval = x.abs().max() / (2 ** (self.bitwidth - 1) - 1) + minimal_num
        return symmsignedquantize(x, self.bitwidth, interval)


class QuantizeConv2d(nn.Conv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.mode = None
        self.quantizer = None
        self.weight_bits = None
        self.weight_scale = None
        self.act_bits = None
        self.x_scale = None
        self.bn_fused = False

    def set_quant_parameters(self, act_bits, weight_bits, quantizer):
        self.act_bits = act_bits
        self.weight_bits = weight_bits
        self.quantizer = quantizer

    def forward(self, x):
        if self.mode == 'raw':
            out = super().forward(x)
        elif self.mode == "quant_forward":
            out = self._quant_conv_forward(x)
        elif self.mode == "calibration_forward":
            out = self.calibrate_forward(x)
        else:
            raise NotImplementedError
        return out

    def _quant_conv_forward(self, x):
        assert self.quantizer.calibrated is not None, f"You should run calibrate_forward before run _quant_conv_forward for {self}"
        weight_sim, bias_sim = self.quantizer.quant_weight_bias(self.weight, self.bias)
        x_sim = self.quantizer.quant_input(x)
        out_sim = F.conv2d(x_sim, weight_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        out_sim = self.quantizer.quant_output(out_sim)
        return out_sim

    def get_quant_weight_bias(self):
        return self.quantizer.quant_weight_bias(self.weight, self.bias)

    def calibrate_forward(self, x):
        # move this into quantizer
        # assert self.weight_bits is not None and self.act_bits is not None, f"You should set the weight_bits and bias_bits for {self}"
        op = lambda input, weight, bias: F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation,
                                                  self.groups)
        out_sim = self.quantizer.calibration(x, self.weight, self.bias, op)
        return out_sim


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
        activation_qbit(int): quantization resolution of the crossbar.
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

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 q_type='uniform',
                 input_qbit=32, weight_qbit=32, activation_qbit=32, scaler_dw=1,
                 is_first_layer=False, is_last_layer=False, **crxb_cfg):

        super(crxb_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        # create new object element for layer notation: add: 21-06-12
        self.q_type = q_type
        if self.q_type in ['robust', 'robust_batch']:
            self.layer_type = 'learnable'
        else:
            self.layer_type = 'fixed'

        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        self.input_qbit = input_qbit if not (self.is_first_layer and input_qbit < 8) else 8
        self.weight_qbit = weight_qbit
        self.activation_qbit = activation_qbit if not (self.is_last_layer and activation_qbit < 8) else 8
        self.calibration = False

        self.quantizer = SimpleCrxbQuantizer(self.weight_qbit, self.input_qbit, 8, self.activation_qbit)

        assert self.groups == 1, "currently not support grouped convolution for custom conv"

        ################## Crossbar conversion #############################
        self.crxb_size = crxb_cfg['crxb_size']
        # 210626 modifidation: data copy optimization
        weight_flatten_rows = self.in_channels * torch.cumprod(torch.tensor(self.kernel_size), 0)[-1].item()
        weight_flatten_cols = self.out_channels
        self.crxb_row, self.crxb_row_pads = num_pad(weight_flatten_rows, self.crxb_size)
        self.crxb_col, self.crxb_col_pads = num_pad(weight_flatten_cols, self.crxb_size)
        # p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        self.w_pad = [0, self.crxb_row_pads, 0, self.crxb_col_pads]
        self.input_pad = [0, 0, 0, self.crxb_row_pads]

        ################# Hardware conversion ##############################
        # weight and input levels
        # ReRAM cells
        # moved to NN_array
        self.phy = NNarray(crxb_row=self.crxb_row, crxb_col=self.crxb_col,
                           output_select=torch.arange(self.out_channels),
                           bias=bias, n_lvl_v=2**input_qbit, **crxb_cfg)

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
        output_crxb = self.phy.crxb_forward(input_crxb, weight_crxb)
        return output_crxb

    def _quant_conv_forward(self, inputs):
        assert self.quantizer.calibrated is not None, f"You should run calibrate_forward before run _quant_conv_forward for {self}"

        # with profiler.record_function("input data and weight quantization"):
        # 1. input data and weight quantization
        w_sim, bias_sim = self.quantizer.quant_weight_bias(self.weight, self.bias)
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

        h_out = int((inputs.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
        w_out = int((inputs.shape[3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)

        # (N, C, H, W)
        output = output.view(output.shape[0:2] + (h_out, w_out))
        return output

    def _reset_delta(self):
        self.delta_in_sum.data[0] = 0
        self.delta_out_sum.data[0] = 0
        self.counter.data[0] = 0
        self.delta_w.data[0] = 1.

    # replace __repr__ with extra_repr
    def extra_repr(self) -> str:
        return super(crxb_Conv2d, self).extra_repr() + \
               "(I, W, A qbit:{} {} {})".format(
                   self.input_qbit if self.input_qbit < 32 else None,
                   self.weight_qbit if self.input_qbit < 32 else None,
                   self.activation_qbit if self.input_qbit < 32 else None)

    def map(self, inputs=None, weight_i=None):
        return_list = []
        if inputs is not None:
            # 2.1 flatten and unfold the weight and input
            input_unfold = F.unfold(inputs, kernel_size=self.kernel_size,
                                    dilation=self.dilation, padding=self.padding,
                                    stride=self.stride)

            # 2.2. add paddings
            input_padded = F.pad(input_unfold, self.input_pad,
                                 mode='constant', value=0)
            # 2.3. reshape to crxb size
            input_crxb = input_padded.view(inputs.shape[0], 1, self.crxb_row,
                                           self.crxb_size, input_padded.shape[2])
            return_list.append(input_crxb)
        if weight_i is not None:
            # 2.1 flatten and unfold the weight and input
            weight_flatten = weight_i.view(self.out_channels, -1)

            # 2.2. add paddings
            weight_padded = F.pad(weight_flatten, self.w_pad,
                                  mode='constant', value=0)
            # 2.3. reshape to crxb size
            weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                             self.crxb_row, self.crxb_size).transpose(1, 2)
            return_list.append(weight_crxb)

        if len(return_list) == 2:
            return return_list
        elif len(return_list) == 1:
            return return_list[0]
        else:
            pass

    # 20210321: add: demap to accommedate the SAF-prune
    def demap(self, weight_crxb):
        # 2.3. reshape to crxb size
        weight_padded = weight_crxb.transpose(1, 2).view(self.crxb_col * self.crxb_size, self.crxb_row * self.crxb_size)
        # 2.2. add paddings
        weight_flatten = weight_padded[self.w_pad[2]:-self.w_pad[3], self.w_pad[0]:-self.w_pad[1]]
        # 2.1 flatten and unfold the weight and input
        weight_i = weight_flatten.view(self.out_channels, self.in_channels, *self.kernel_size)
        return weight_i

    def w_quantizer_update(self):
        if self.q_type in ['robust', 'robust_batch']:
            # 210626 data copy optimization
            X = (self.weight.detach() / self.delta_w)  # .clone()
            # add clip operation
            # X = torch.clamp(X, -1, 1)
            X.clamp_(-1, 1)
            self.w_quantizer.fit(X)
        else:
            pass

    def calibrate_forward(self, inputs):
        quant_i, x_sim, w_sim = self.quantizer.calibration(inputs, self.weight, self.bias, self.op1)
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

        h_out = int((inputs.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
        w_out = int((inputs.shape[3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)

        # (N, C, H, W)
        output = output.view(output.shape[0:2] + (h_out, w_out))
        return output


if __name__ == "__main__":
    cfg = (64, 32, 3)
    conv_torch = nn.Conv2d(*cfg).cuda()
    feature = torch.rand(1, 64, 28, 28).cuda()
    crxb_cfg = {'gmax': 0.000333, 'gmin': 3.33e-07, 'gwire': 0.0357, 'gload': 0.25, 'q_type': 'robust_batch',
                'input_qbit': 8, 'weight_qbit': 8, 'activation_qbit': 8, 'vdd': 3.3, 'freq': 10000000.0, 'temp': 300,
                'crxb_size': 64, 'enable_SAF': False, 'enable_ec_SAF': False, 'enable_noise': False, 'noise_amp': 0.12,
                'random_noise': True, 'ir_drop': False, 'device': 'cuda', 'enable_ec_Var': False, 'p_fix': 0.1}
    conv_crxb = crxb_Conv2d(*cfg, **crxb_cfg).cuda()
    def model_conversion(model_ref: nn.Module, model_out: nn.Module, calibration_inputs):
        model_dict = model_out.state_dict()
        pretrained_dict = model_ref.state_dict()
        for k, v in model_dict.items():
            if k in pretrained_dict:
                model_dict[k].copy_(pretrained_dict[k].data)
        model_out.mode = "calibration_forward"
        model_out(calibration_inputs)
        model_out.mode = "quant_forward"

    model_conversion(conv_torch, conv_crxb, feature)
    ref_o = conv_torch(feature)
    q_o = conv_crxb(feature)
    print((q_o - ref_o).abs().max())
    diff = F.mse_loss(q_o, ref_o, reduction="sum")
    print(diff)
