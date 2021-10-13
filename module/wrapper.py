import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.nn.modules import module

from .layer_q import crxb_Conv2d
from .quantization.layer_quantizer import RobustCrxbQuantizer, SimpleCrxbQuantizer

crxb_cfg = {'gmax': 0.000333, 'gmin': 3.33e-07, 'gwire': 0.0357, 'gload': 0.25, 'q_type': 'robust_batch',
            'input_qbit': 8, 'weight_qbit': 8, 'inter_activation_qbit': 8, 'vdd': 3.3, 'freq': 10000000.0, 'temp': 300,
            'crxb_size': 64, 'enable_SAF': False, 'enable_ec_SAF': False, 'enable_noise': False, 'noise_amp': 0.12,
            'random_noise': True, 'ir_drop': False, 'device': 'cuda', 'enable_ec_Var': False, 'p_fix': 0.1}

def wrap_modules_in_net(net,layer_quantizer=RobustCrxbQuantizer,quantizer_kwargs={},conv_wrapper=SimpleCrxbQuantizer,wrap_fc=False):
    wrapped_modules={}
    module_dict=OrderedDict()
    for name,m in net.named_modules():
        module_dict[name]=m
        if isinstance(m,(nn.Conv2d,nn.Linear) if wrap_fc else nn.Conv2d):
            if isinstance(m,nn.Conv2d):
                cfg = {
                    "in_channels" : m.in_channels, 
                    "out_channels" : m.out_channels, 
                    "kernel_size" : m.kernel_size, 
                    "stride" : m.stride, 
                    "padding" : m.padding, 
                    "dilation" : m.dilation, 
                    "transposed" : m.transposed, 
                    "output_padding" : m.output_padding, 
                    "groups" : m.groups, 
                    "bias" : m.bias is not None,
                    "padding_mode" : m.padding_mode, 
                }
                conv_crxb = crxb_Conv2d(**cfg, **crxb_cfg)
                new_m=conv_crxb
                new_m.weight.data=m.weight.data
                new_m.bias=m.bias
            elif isinstance(m,nn.Linear):
                new_m = m
            new_m.mode='raw'
            if isinstance(layer_quantizer, RobustCrxbQuantizer):
                if not quantizer_kwargs.get("gamma"):
                    quantizer_kwargs.update({"gamma":0.})
            conv_crxb.quantizer = layer_quantizer(conv_crxb.weight_qbit, conv_crxb.input_qbit, 8, **quantizer_kwargs)
            wrapped_modules[name]=new_m
    return wrapped_modules

def model_conversion(model_ref: nn.Module, model_out: nn.Module, calibration_inputs):
    model_dict = model_out.state_dict()
    pretrained_dict = model_ref.state_dict()
    for k, v in model_dict.items():
        if k in pretrained_dict:
            model_dict[k].copy_(pretrained_dict[k].data)
    model_out.mode = "calibration_forward"
    model_out(*calibration_inputs)
    model_out.mode = "quant_forward"
