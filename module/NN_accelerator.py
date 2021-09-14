import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .w2g import x_relu

__all__ = ['NNarray', 'num_pad']


def num_pad(source, target):
    crxb_index = math.ceil(source / target)
    num_padding = crxb_index * target - source
    return crxb_index, num_padding


class NNarray(nn.Module):
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

    def __init__(self, ir_drop, device, gmax, gmin, gwire, gload,
                 crxb_row=16, crxb_col=16, output_select=torch.arange(16*64),
                 #  bias=True,
                 input_qbit=32, weight_qbit=32, activation_qbit=32, scaler_dy=1,
                 n_lvl_v=2**8,
                 #  is_first_layer=False, is_last_layer=False,
                 vdd=3.3, enable_noise=True, freq=10e6, temp=300, crxb_size=64,
                 enable_SAF=False, enable_ec_SAF=False, enable_rand=False,
                 noise_amp=0.12, noise_mean=None, noise_var=None, random_noise=False,
                 # 20201013: -Fix bug: fix mean shift after noise injection
                 # 20210319: add: row_selection: part A
                 enable_ec_Var=False, p_fix=0.1, input_select=torch.arange(16*64), **kwargs):

        super(NNarray, self).__init__()
        self.ir_drop = ir_drop
        self.device = device

        ################## Crossbar conversion #############################
        self.crxb_size = crxb_size
        self.enable_ec_SAF = enable_ec_SAF

        # 20210319: add: row_selection: part A
        self.register_buffer('nchin_index', input_select)
        self.register_buffer('nchout_index', output_select)
        self.crxb_row = crxb_row
        self.crxb_col = crxb_col
        weight_crxb_shape = torch.Size((self.crxb_col, self.crxb_row,
                                        self.crxb_size, self.crxb_size))
        # self.weight = nn.Parameter(torch.Tensor(weight_crxb_shape))

        ################# Hardware conversion ##############################
        # weight and input levels
        self.n_lvl_v = n_lvl_v
        self.Gmax = gmax  # max conductance
        self.Gmin = gmin  # min conductance

        ################# weight conversion ##############################
        # self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA0=self.Gmax,
        #                G_SA1=self.Gmin, weight_shape=weight_crxb_shape, enable_SAF=False)
        self.delta_g = self.Gmax - self.Gmin
        self.G_SA0 = self.Gmax
        self.G_SA1 = self.Gmin

        if enable_SAF:
            from .SAF import SAF
            # use only one SAF module
            self.SAF = SAF(weight_crxb_shape, p_SA0=self.p_SA0, p_SA1=self.p_SA1,
                           G_SA0=self.G_SA0, G_SA1=self.G_SA1)
        else:
            self.SAF = None

        if enable_noise:
            from .variation import Variation
            self.Var = Variation(weight_crxb_shape, noise_amp, random_noise, noise_mean, noise_var)
        else:
            self.Var = None
        self.enable_ec_Var = enable_ec_Var
        self.p_fix = p_fix

        self.register_buffer("G_crxb", torch.full(torch.Size( (2,)) + weight_crxb_shape, self.Gmin, device=self.device))
        self.Gwire = gwire
        ################################# DAC/ADC #######################################
        self.Gload = gload
        self.Vdd = vdd  # unit: volt
        self.delta_v = self.Vdd / (self.n_lvl_v - 1)

        ################ Stochastic Conductance Noise setup #########################
        # parameters setup
        self.enable_noise = enable_noise
        self.freq = freq  # operating frequency
        self.kb = 1.38e-23  # Boltzmann const
        self.temp = temp  # temperature in kelvin
        self.q = 1.6e-19  # electron charge

        self.tau = 0.5  # Probability of RTN
        self.a = 1.662e-7  # RTN fitting parameter
        self.b = 0.0015  # RTN fitting parameter

        self.noise_amp = noise_amp
        self.noise_mean = noise_mean
        # 20201013: -Fix bug: fix mean shift after noise injection
        self.noise_var = noise_var
        self.random_noise = random_noise

        self.p_SA0 = 0.1
        self.p_SA1 = 0.1
        self.enable_rand = enable_rand
        self.enable_SAF = enable_SAF

    def crxb_forward(self, inputs, weight):

        input_crxb = inputs * self.delta_v  # convert to voltage
        # 2. Perform the computation between input voltage and weight conductance
        # convert the floating point weight into conductance pair values
        G_crxb_i = self.w2g(weight)

        # this block is for introducing stochastic noise into ReRAM conductance
        if self.enable_SAF:
            G_crxb = self.SAF(G_crxb_i)
        else:
            G_crxb = G_crxb_i
        if self.enable_noise:  # 20201008-fix: noise with < 0 constrain!! for determined noise distribution
            # print(G_crxb.norm(1))
            G_crxb = self.Var(G_crxb)
            # print("pp",G_crxb.norm(1))
            # if self.enable_ec_Var:
            #     G_crxb -= G_crxb_i.detach() * math.exp((abs(self.noise_amp / 3) ** 2) / 2) * self.p_fix
        self.G_crxb.data = G_crxb.data

        # 2.4. compute matrix multiplication
        # this block is to calculate the ir drop of the crossbar

        if self.ir_drop:
            from .IR_solver import IrSolver

            crxb_pos = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[0].permute(3, 2, 0, 1),
                                device=self.device)
            crxb_pos.resetcoo()
            crxb_neg = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[1].permute(3, 2, 0, 1),
                                device=self.device)
            crxb_neg.resetcoo()

            output_crxb = (crxb_pos.caliout() - crxb_neg.caliout())
            output_crxb = output_crxb.contiguous().view(self.crxb_col,
                                                        self.crxb_row,
                                                        self.crxb_size,
                                                        inputs.shape[0],
                                                        inputs.shape[-1])

            # (N, crxb_col, crxb_size, crxb_row, L)
            output_crxb = output_crxb.permute(3, 0, 2, 1, 4)

        else:
            # (N, crxb_col, crxb_row, crxb_size, L)
            output_crxb = torch.matmul(G_crxb[0], input_crxb) - \
                torch.matmul(G_crxb[1], input_crxb)
            # (N, crxb_col, crxb_size, crxb_row, L)
            output_crxb.transpose_(2, 3)
            # print(output_crxb.shape)

        # 3. perform ADC operation (i.e., current to digital conversion)
        output_shape = (output_crxb.shape[0], output_crxb.shape[1] * output_crxb.shape[2],
                        output_crxb.shape[3], output_crxb.shape[4])
        output = output_crxb.reshape(output_shape).index_select(
            dim=1, index=self.nchout_index)
        # print(output.shape)
        # (N, C, crxb_row, L) or (N, C, crxb_row)
        return output.squeeze(-1)

    def extra_repr(self):
        return "{}x{},Voltage:{}V,{}-lvls,G:{}-{})".format(self.crxb_row, self.crxb_col,
                                                           self.n_lvl_v, self.Vdd,
                                                           self.Gmin, self.Gmax)

    def swv(self, weight, out_index: torch.LongTensor = None, in_index: torch.LongTensor = None) -> float:
        # convert the floating point weight into conductance pair values
        G_crxb = self.w2g(weight)
        # MARK: change the sequence of the rows and columns
        if out_index is not None or in_index is not None:
            G_crxb = self.map_matrix_transform(G_crxb, out_index, in_index)
        # this block is for introducing stochastic noise into ReRAM conductance
        print(G_crxb.shape)
        if self.enable_SAF:
            G_crxb = self.SAF(G_crxb)
        if self.enable_noise and self.noise_amp > 0.:
            G_crxb = self.Var(G_crxb)
        return self.w2g(weight) - G_crxb

    def w2g(self, input):
        # x_relu() function is Critical
        G_pos = self.Gmin + x_relu(input) * self.delta_g
        G_neg = self.Gmin + F.relu(-input) * self.delta_g
        return torch.stack((G_pos, G_neg), 0)

    def partial_acc(self, output_adc):
        output_sum = torch.sum(output_adc, dim=2)
        return output_sum

    def g2w(self, input):
        return (input[0] - input[1]) / self.delta_g

    def error_compensation(self) -> Tensor:
        SA0 = self.SAF.index_SA0().float().cuda()
        SA1 = self.SAF.index_SA1().float().cuda()
        G_diff = (self.G_crxb - self.G_SA0) * SA0 + \
                 (self.G_crxb - self.G_SA1) * SA1
        return G_diff

    def update_SAF(self, enable_SAF, p_SA0, p_SA1, new_SAF_mask=False, enable_rand=False):
        self.p_SA0 = p_SA0
        self.p_SA1 = p_SA1
        self.enable_SAF = enable_SAF
        # update the SAF modules
        self.SAF.p_SA0.data[0] = self.p_SA0
        self.SAF.p_SA1.data[0] = self.p_SA1
        # enable the random mask, thus each forward call get a new p_state mask
        self.enable_rand = enable_rand
        self.SAF.enable_rand = enable_rand

        if new_SAF_mask:
            self.SAF.p_state.data.uniform_()

    def update_variation(self, noise_amp, random_noise, noise_mean=None, noise_var=None, dist='normal',
                         # 20201013: -Fix bug: fix mean shift after noise injection
                         p_fix=None, enable_ec_Var=None):
        self.noise_amp = noise_amp
        self.noise_mean = noise_mean
        self.noise_var = noise_var
        self.random_noise = random_noise
        self.Var.update_variation_profile(
            noise_amp, random_noise, noise_mean, noise_var, dist)
        # 20201013: -Fix bug: fix mean shift after noise injection
        if enable_ec_Var is not None:
            self.enable_ec_Var = enable_ec_Var
        if p_fix is not None:
            self.p_fix = p_fix

    # 20201013: -Fix bug: fix mean shift after noise injection
    def var_comp(self, conductance, p_fix=None):
        if p_fix is None:
            p_fix = self.p_fix
        # use 2* \sigma^2 because there are two array \mu - \mu + (2* \sigma)^2/2
        if self.noise_mean is not None and self.noise_var is not None:
            factor = math.exp((self.noise_mean + self.noise_var ** 2) / 2)
        else:
            factor = math.exp((abs(self.noise_amp / 3) ** 2) / 2)
        outputs = (conductance[0]-conductance[1]
                   ).sum(dim=(1, 3)) * factor * p_fix * self.Vdd
        return outputs.view(-1).index_select(dim=0, index=self.nchout_index)

    def map_matrix_transform(self, input_m: torch.Tensor, transform_o: torch.LongTensor, transform_i: torch.LongTensor,
                             mode="derivable"):
        flatten = input_m.transpose(1, 2).view(
            self.crxb_col * self.crxb_size, self.crxb_row * self.crxb_size)
        if transform_i is not None:
            assert transform_i.shape == torch.Size(
                (self.crxb_col * self.crxb_size,))
            if mode == "derivable":
                row_tran = torch.eye(self.crxb_row * self.crxb_size,
                                     device=self.device).index_select(dim=0, index=transform_i)
                swap = torch.matmul(flatten, row_tran)
            else:
                swap = flatten.data.index_select(1, transform_i)
        else:
            swap = flatten

        if transform_o is not None:
            assert transform_o.shape == torch.Size(
                (self.crxb_col * self.crxb_size,))
            if mode == "derivable":
                col_tran = torch.eye(self.crxb_col * self.crxb_size,
                                     device=self.device).index_select(dim=0, index=transform_o)
                swap = torch.matmul(col_tran, swap)
            else:
                swap = swap.data.index_select(0, transform_o)
        return swap.view(self.crxb_col, self.crxb_size, self.crxb_row, self.crxb_size).transpose(1, 2)

# Change list:
# 2020/03/15: fixed the bias bug
# 2020/03/25: divide the quatization bits into three class, IA, W, MA.
# 2020/03/25: add layer number flag: is_first_layer, is_last_layer
# 2020/05/22: change input all to inputs, may cause check fail in some version
# 2020/07/05: -add: use quantiztion or not option, overload __repr__ to show qbit
#             -fix:  self.h_lvl_ma = 2 ** (input_qbit - 1) - 1 if not self.is_first_layer.data else 2 ** (8 - 1) - 1
#                    self.h_lvl_w = 2 ** (weight_qbit - 1) - 1
#                    self.h_lvl_ia = 2 ** (activation_qbit - 1) - 1 if not self.is_last_layer.data else 2 ** (8 - 1) - 1
# 2020/07/08: -fix: quantized weight and delta_w buffer can be updated and stored now!
