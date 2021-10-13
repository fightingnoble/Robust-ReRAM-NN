import argparse
import random
import os
import copy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
# import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import os,sys
os.chdir(sys.path[0])
sys.path.insert(0,'..')
# print(sys.path)

import datasets
from module.layer_q import crxb_Conv2d
from module.quantization.layer_quantizer import RobustCrxbQuantizer, SimpleCrxbQuantizer


vgg11 = models.vgg11(pretrained=True)


def model_conversion(model_ref: nn.Module, model_out: nn.Module, calibration_inputs):
    model_dict = model_out.state_dict()
    pretrained_dict = model_ref.state_dict()
    for k, v in model_dict.items():
        if k in pretrained_dict:
            model_dict[k].copy_(pretrained_dict[k].data)
    model_out.mode = "calibration_forward"
    model_out(*calibration_inputs)
    model_out.mode = "qat_forward"

crxb_cfg = {'gmax': 0.000333, 'gmin': 3.33e-07, 'gwire': 0.0357, 'gload': 0.25, 'q_type': 'robust_batch',
            'input_qbit': 8, 'weight_qbit': 8, 'inter_activation_qbit': 8, 'vdd': 3.3, 'freq': 10000000.0, 'temp': 300,
            'crxb_size': 64, 'enable_SAF': False, 'enable_ec_SAF': False, 'enable_noise': False, 'noise_amp': 0.12,
            'random_noise': True, 'ir_drop': False, 'device': 'cuda', 'enable_ec_Var': False, 'p_fix': 0.1}


# print(list(vgg11.features.children()))

test_m = list(vgg11.features.children())[0]
print(test_m)

def get_data():
    parser = argparse.ArgumentParser(description='Imagenet test (and eval) a model')
    parser.add_argument('--data', type=str, default='/datasets/imagenet', 
                        help="path of imagenet datasets.")
    parser.add_argument('--calib_size', default=32,type=int)
    parser.add_argument('--quant_granularity', type=str, default='', 
                        choices=['channelwise', 'layerwise', 'channelwise_weightonly', 'crossbarwise'], 
                        help="choose the quantization granularity.")
                        
    parser.add_argument('--w_bits',type=int,default=8)
    parser.add_argument('--a_bits',type=int,default=8)
    parser.add_argument('--quantizer', type=str, default='', 
                        choices=['easyquant', 'pow2_easyquant', 'quantile', 'adaquant'], 
                        help="choose a quantization mode.")

    parser.add_argument('--model', type=str, default='', 
                        choices=['resnet18', 'resnet50', 'vgg16'], 
                        help="choose a model.")

    parser.add_argument('--batch_norm_tuning',action='store_true', default=False,
                    help='switch to turn on batch norm tuning for AdaQuant')
    parser.add_argument('--bias_tuning',action='store_true', default=False,
                    help='switch to turn on bias tuning for AdaQuant')
    parser.add_argument('--sequential',action='store_true', default=False,
                    help='switch to turn on sequential mode for AdaQuant')
    parser.add_argument('--seed', default=0,type=int)

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--distributed',action='store_true', default=False,
                    help='distributed training')

    parser.add_argument('--opt', type=str, default='Adam', 
                        choices=['Adam', 'SGD'], 
                        help="choose a optimizer.")
    parser.add_argument('--lr', default=1,type=float)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    # enhance reproductivity
    cudnn.benchmark = False
    np.random.seed(args.seed)

    cudnn.benchmark = True

    # Data loading code
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    val_dataset = datasets.ImageFolder(valdir, val_transformer)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    inds=np.random.permutation(len(train_dataset))[:1024]
    calibset = torch.utils.data.Subset(copy.deepcopy(train_dataset),inds)
    calibset.transform = val_transformer
    calib_loader = torch.utils.data.DataLoader(
        calibset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Data loading code

    g=datasets.ImageNetLoaderGenerator('/datasets/imagenet','imagenet',args.calib_size,32,8)
    val_loader=g.test_loader()
    calib_loader=g.calib_loader(num=1024)
    return val_loader, calib_loader

def test_classification(net, test_loader):
    pos=0
    tot=0
    with torch.no_grad():
        q=tqdm(test_loader)
        for inp,target in q:
            inp,target=inp.cuda(), target.cuda()
            out=net(inp)
            pos_num=torch.sum(out.argmax(1)==target).item()
            pos+=pos_num
            tot+=inp.size(0)
            q.set_postfix({"acc":pos/tot})
    print(pos/tot)


val_loader, calib_loader = get_data()
for gamma_t in range(10):
    gamma = gamma_t *10
    def crxb_conversion(module: nn.Module, inputs):
        cfg = {
            "in_channels" : module.in_channels, 
            "out_channels" : module.out_channels, 
            "kernel_size" : module.kernel_size, 
            "stride" : module.stride, 
            "padding" : module.padding, 
            "dilation" : module.dilation, 
            "transposed" : module.transposed, 
            "output_padding" : module.output_padding, 
            "groups" : module.groups, 
            "bias" : module.bias is not None,
            "padding_mode" : module.padding_mode, 
        }
        conv_crxb = crxb_Conv2d(**cfg, **crxb_cfg)
        conv_crxb.quantizer = SimpleCrxbQuantizer(conv_crxb.weight_qbit, conv_crxb.input_qbit, 8, )
        # conv_crxb.quantizer = RobustCrxbQuantizer(conv_crxb.weight_qbit, conv_crxb.input_qbit, 8, gamma=gamma_t)
        conv_crxb.cuda()
        model_conversion(module, conv_crxb, inputs)

        ref_o = module.forward(*inputs)
        q_o = conv_crxb.forward(*inputs)
        print("\nmax: {}\n".format((q_o - ref_o).abs().max()))
        diff = F.mse_loss(q_o, ref_o, reduction="sum")
        print("\ntotal: {}\n".format(diff))

        module, module_bk = conv_crxb, module
        module.bk = module_bk
        # print(callable(module))

    hook = test_m.register_forward_pre_hook(crxb_conversion)
    vgg11.cuda()
    test_classification(vgg11, val_loader)

    # # print(model, "\n\n\n\n",list(model.children()))
    # # assert False
    # test_m = list(model.backbone.features.children())[0].conv
    # print(test_m)

    # hooks = []
    # for gamma_t in range(10):
    #     gamma = gamma_t *0.1
    #     def crxb_conversion(module: nn.Module, inputs):
    #         cfg = {
    #             "in_channels" : module.in_channels, 
    #             "out_channels" : module.out_channels, 
    #             "kernel_size" : module.kernel_size, 
    #             "stride" : module.stride, 
    #             "padding" : module.padding, 
    #             "dilation" : module.dilation, 
    #             "transposed" : module.transposed, 
    #             "output_padding" : module.output_padding, 
    #             "groups" : module.groups, 
    #             "bias" : module.bias is not None,
    #             "padding_mode" : module.padding_mode, 
    #         }
    #         conv_crxb = crxb_Conv2d(**cfg, **crxb_cfg)
    #         # conv_crxb.quantizer = SimpleCrxbQuantizer(conv_crxb.weight_qbit, conv_crxb.input_qbit, 8, )
    #         conv_crxb.quantizer = RobustCrxbQuantizer(conv_crxb.weight_qbit, conv_crxb.input_qbit, 8, gamma=gamma)
    #         conv_crxb.cuda()
    #         model_conversion(module, conv_crxb, inputs)

    #         ref_o = module.forward(*inputs)
    #         q_o = conv_crxb.forward(*inputs)
    #         print("max: {}\n".format((q_o - ref_o).abs().max()))
    #         diff = F.mse_loss(q_o, ref_o, reduction="sum")
    #         print("total: {}\n".format(diff))

    #         module, module_bk = conv_crxb, module
    #         module.bk = module_bk
    #         # print(callable(module))

    #     hook = test_m.register_forward_pre_hook(crxb_conversion)
    #     # hooks.append(hook)
    #     print("\ncalibration\n")
    #     test_model(distributed, args, model, calib_loader, calib_dataset, CLASSES)
    #     # if hooks is not None:
    #     #     for hook in hooks:
    #     #         hook.remove()
