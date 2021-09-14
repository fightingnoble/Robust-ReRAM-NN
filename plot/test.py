import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models

import os,sys
os.chdir(sys.path[0])
sys.path.insert(0,'..')
# print(sys.path)
from module.layer_q import crxb_Conv2d
from module.quantization.layer_quantizer import RobustCrxbQuantizer

crxb_cfg = {'gmax': 0.000333, 'gmin': 3.33e-07, 'gwire': 0.0357, 'gload': 0.25, 'q_type': 'robust_batch',
            'input_qbit': 8, 'weight_qbit': 8, 'activation_qbit': 8, 'vdd': 3.3, 'freq': 10000000.0, 'temp': 300,
            'crxb_size': 64, 'enable_SAF': False, 'enable_ec_SAF': False, 'enable_noise': False, 'noise_amp': 0.12,
            'random_noise': True, 'ir_drop': False, 'device': 'cuda', 'enable_ec_Var': False, 'p_fix': 0.1}

if __name__ == "__main__":
    # cfg = (3, 64, 3, 1, 1)
    # conv_torch = nn.Conv2d(*cfg).cuda()

    # # print(dir(conv_torch),"\n*****\n", vars(conv_torch))
    # conv_crxb = crxb_Conv2d(*cfg, **crxb_cfg)

    vgg11 = models.vgg11(pretrained=True).cuda()
    conv_torch = list(vgg11.features.children())[0]
    cfg = {
            "in_channels" : conv_torch.in_channels, 
            "out_channels" : conv_torch.out_channels, 
            "kernel_size" : conv_torch.kernel_size, 
            "stride" : conv_torch.stride, 
            "padding" : conv_torch.padding, 
            "dilation" : conv_torch.dilation, 
            "transposed" : conv_torch.transposed, 
            "output_padding" : conv_torch.output_padding, 
            "groups" : conv_torch.groups, 
            "bias" : conv_torch.bias is not None,
            "padding_mode" : conv_torch.padding_mode, 
        }
    conv_crxb = crxb_Conv2d(**cfg, **crxb_cfg)

    feature = torch.rand(1, 3, 224, 224).cuda()

    print(conv_torch, conv_crxb)
    # print(dir(conv_crxb),"\n!!!!!!\n", vars(conv_torch))
    conv_crxb.quantizer = RobustCrxbQuantizer(conv_crxb.weight_qbit, conv_crxb.input_qbit, 8, gamma=0.)
    conv_crxb.cuda()
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
