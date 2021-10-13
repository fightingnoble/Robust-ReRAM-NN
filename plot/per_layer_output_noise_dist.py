# Copyright (c) OpenMMLab. All rights reserved.

# add by zhangchg
from typing import List, OrderedDict, AnyStr, Union

import argparse
from collections import OrderedDict
import os
import warnings

import mmcv
import numpy as np
import torch

# added by zhangchg
import torch.nn as nn
import torch.nn.functional as F

from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

# zhangchg modified
# from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.apis.test import collect_results_cpu, collect_results_gpu
import os.path as osp
import time
from mmcv.image import tensor2imgs
from tqdm import tqdm

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model

import os,sys
os.chdir(sys.path[0])
sys.path.insert(0,'..')
from module.layer_q import crxb_Conv2d, crxb_Linear, convert, convert_weight, reset_model
from module.quantization.layer_quantizer import RobustCrxbQuantizer, SimpleCrxbQuantizer, MinibatchRobustCrxbQuantizer
from module.layer_q import QuantizedLayer

import pandas as pd
import csv

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
            'input_qbit': 8, 'weight_qbit': 4, 'inter_activation_qbit': 8, 'vdd': 3.3, 'freq': 10000000.0, 'temp': 300,
            'crxb_size': 64, 'enable_SAF': False, 'enable_ec_SAF': False, 'enable_noise': False, 'noise_amp': 0.8,
            'random_noise': True, 'ir_drop': False, 'device': 'cuda', 'enable_ec_Var': False, 'p_fix': 0.1}


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')

    parser.add_argument('--calib_size', default=32,type=int)

    parser.add_argument('--w_bits',type=int,default=8)
    parser.add_argument('--a_bits',type=int,default=8)
    parser.add_argument('--quantizer', type=str, default='RobustCrxbQuantizer', 
                        choices=['RobustCrxbQuantizer', 'SimpleCrxbQuantizer', 'MinibatchRobustCrxbQuantizer'], 
                        help="choose a quantization mode.")

    # noise test
    # 20201008-add: add number controling for random running for training/noise test
    # 20210313:comment: clarify the variable is used for disable/control times of monte carlo in training or noise test
    # To enable the monte-carlo simulation, one should set random-ite>1 and enable the random noise.
    # If noise test is performed, enable-noise/enable_SAF also should be enabled

    # noise test parameter
    parser.add_argument('--random-ite', default=20, type=int,
                        help='Number of Monte Carlo simulations for training/noise test')
    parser.add_argument('--noise-amp', type=float, default=0.5,
                        help='variation amp of conductance')
    parser.add_argument('--step', type=float, default=0.05,
                        help='step for variation amp of conductance sweeping')
    parser.add_argument('--noise-sweep', action='store_true', default=False,
                        help='enable robustness analysis w.r.t. different var(iation) amp of conductance '
                            'by sweeping the noise-amp')
    # variation parameter
    parser.add_argument('--enable-noise', action='store_true', default=False,
                        help='switch to turn on conductance variation(var injection)')
    parser.add_argument('--random-noise', action='store_true', default=True,
                        help='switch to turn on conductance variation random distribution')
    # SAF parameter
    parser.add_argument('--enable_SAF', action='store_true', default=False,
                        help='switch to turn on SAF analysis')
    parser.add_argument('--enable-ec-SAF', action='store_true', default=False,
                        help='switch to turn on SAF error correction')
    # ir-drop parameter
    parser.add_argument('--ir-drop', action='store_true', default=False,
                    help='switch to turn on ir drop analysis')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    **show_kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # print('\t',id(model.state_dict()['module.backbone.features.0.quantizer.o_quantizer.sum']))
            result = model(return_loss=False, **data)

        batch_size = len(result)
        results.extend(result)

        if show or out_dir:
            scores = np.vstack(result)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [model.CLASSES[lb] for lb in pred_label]

            img_metas = data['img_metas'].data[0]
            imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                result_show = {
                    'pred_score': pred_score[i],
                    'pred_label': pred_label[i],
                    'pred_class': pred_class[i]
                }
                model.module.show_result(
                    img_show,
                    result_show,
                    show=show,
                    out_file=out_file,
                    **show_kwargs)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError((f'The tmpdir {tmpdir} already exists.',
                           ' Since tmpdir will be deleted after testing,',
                           ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def test_model(distributed, args, model, data_loader, dataset, CLASSES):
    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[0])
        model.CLASSES = CLASSES
        show_kwargs = {} if args.show_options is None else args.show_options
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        if args.metrics:
            eval_results = dataset.evaluate(outputs, args.metrics,
                                            args.metric_options)
            results.update(eval_results)
            for k, v in eval_results.items():
                print(f'\n{k} : {v:.2f}\n')
        if args.out:
            if 'none' not in args.out_items:
                scores = np.vstack(outputs)
                pred_score = np.max(scores, axis=1)
                pred_label = np.argmax(scores, axis=1)
                pred_class = [CLASSES[lb] for lb in pred_label]
                res_items = {
                    'class_scores': scores,
                    'pred_score': pred_score,
                    'pred_label': pred_label,
                    'pred_class': pred_class
                }
                if 'all' in args.out_items:
                    results.update(res_items)
                else:
                    for key in args.out_items:
                        results[key] = res_items[key]
            print(f'\ndumping results to {args.out}')
            mmcv.dump(results, args.out)


def calib_model(distributed, args, model:nn.Module, 
                wrapped_modules, 
                data_loader, dataset, CLASSES, add_calib_hook=False, sequential=False):
    
    hooks = [] if add_calib_hook else None
    for name,module in wrapped_modules.items():
        module.mode='calibration_forward'
        if add_calib_hook: 
            hook = module.add_calib_hook(sequential)
            hooks.append(hook)
        
    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[0])
        model.CLASSES = CLASSES
        show_kwargs = {} if args.show_options is None else args.show_options
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    for name,module in model.named_modules():
        # if name in wrapped_modules.keys():
        if module in wrapped_modules.values():
            print(f"{name}: {module.quantizer}")
            module.mode='qat_forward'

    if hooks is not None:
        for hook in hooks:
            hook.remove()
    print("calibration finished")

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        # if args.metrics:
        eval_results = dataset.evaluate(outputs, 'accuracy', # args.metrics,
                                        args.metric_options
                                        )
        results.update(eval_results)
        k,v = list(eval_results.items())[0]
        print(k, v)
        return v
        # for k, v in eval_results.items():
        #     print(f'\n{k} : {v:.2f}\n')
        # if args.out:
        #     if 'none' not in args.out_items:
        #         scores = np.vstack(outputs)
        #         pred_score = np.max(scores, axis=1)
        #         pred_label = np.argmax(scores, axis=1)
        #         pred_class = [CLASSES[lb] for lb in pred_label]
        #         res_items = {
        #             'class_scores': scores,
        #             'pred_score': pred_score,
        #             'pred_label': pred_label,
        #             'pred_class': pred_class
        #         }
        #         if 'all' in args.out_items:
        #             results.update(res_items)
        #         else:
        #             for key in args.out_items:
        #                 results[key] = res_items[key]
        #     print(f'\ndumping results to {args.out}')
        #     mmcv.dump(results, args.out)


# 20201008-modified: adapt new NN_array layer
def update_noise(args, model, enable_SAF, new_SAF_mask, noise_amp, random_noise, p_SA0=0.1, p_SA1=0.1):
    n = 0
    for m in model.modules():
        if isinstance(m, (crxb_Linear, crxb_Conv2d)):
            n = n + 1
            m.phy.update_variation(args.enable_noise, noise_amp, random_noise, noise_mean=0.)
            m.phy.update_SAF(args.enable_SAF, p_SA0, p_SA1, new_SAF_mask, enable_rand=False)

    if args.enable_noise:
        print("{}% noise is injected into {} layers".format(noise_amp * 100, n))
    if args.enable_SAF:
        print("SAF-{}SA0-{}SA1 is injected into {} layers".format(p_SA0, p_SA1, n))


def test_model_acc(distributed, args, model:nn.Module, data_loader, dataset, CLASSES): 
    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[0])
        model.CLASSES = CLASSES
        show_kwargs = {} if args.show_options is None else args.show_options
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        eval_results = dataset.evaluate(outputs, 'accuracy', # args.metrics,
                                        args.metric_options
                                        )
        results.update(eval_results)
        k,v = list(eval_results.items())[0]
        print(k, v)
        return v


# 20210313-rename-begin: rename to clarify the function is only used for sweep test not for other test.
def noise_sweep_test(name:str, distributed, args, model:nn.Module, data_loader, dataset, CLASSES, error_list=[]):
    import csv
    f1 = open("sweep_result_acc1.csv", mode="a", newline='')
    csv_f1 = csv.writer(f1)

    acc_lst = [name]
    step = args.step
    for noise_amp in torch.arange(-args.noise_amp, args.noise_amp + step, step):
        # Do mento-carlo simulation with random seeds
        if args.random_noise:
            seeds = torch.randint(1000, (args.random_ite,))
            acc_lst_t = []

            for rand_seed in seeds:
                torch.manual_seed(rand_seed)
                # 20201008-modified: adapt new NN_array layer
                update_noise(args, model, args.enable_SAF, True, noise_amp, True)
                error_list.append
                acc1 = test_model_acc(distributed, args, model, data_loader, dataset, CLASSES)
                acc_lst_t.append(acc1)
            acc_lst_t = torch.tensor(acc_lst_t)
            acc_lst.append(["{}".format(noise_amp),acc_lst_t.max().item(), acc_lst_t.mean().item(), acc_lst_t.min().item()])
        else:
            acc1 = test_model_acc(distributed, args, model, data_loader, dataset, CLASSES)
            acc_lst.append(acc1)

    csv_f1.writerow(acc_lst)
    f1.close()
    # import numpy as np
    # if args.random_noise:
    #     end_n = "_rand"
    # else:
    #     end_n = ""
    # # 20201007 add: add path for output model
    # test_path = os.path.join(args.checkpoint_path, "test/")
    # PATH = os.path.join(test_path, args.check_path)
    # np.save("{}_acc_lst{}".format(PATH, end_n), np.array(acc_lst))


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)

    # add the calibration loader
    calib_dataset = build_dataset(cfg.data.val)
    np.random.seed(3)
    calib_inds=np.random.randint(len(calib_dataset.data_infos),size=args.calib_size)
    calib_dataset.data_infos=[calib_dataset.data_infos[_] for _ in calib_inds]

    # the extra round_up data will be removed during gpu/cpu collect
    calib_loader = build_dataloader(
        calib_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)
 
    # build the model and load checkpoint
    model:nn.Module = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    wrapped_modules:OrderedDict[AnyStr, QuantizedLayer] = OrderedDict()
    crxb_cfg['inter_activation_qbit'] = args.a_bits
    crxb_cfg['weight_qbit'] = args.w_bits
    crxb_cfg['input_qbit'] = args.a_bits
    # RobustCrxbQuantizer, SimpleCrxbQuantizer, MinibatchRobustCrxbQuantizer
    if args.quantizer == 'RobustCrxbQuantizer':
        layer_quantizer=RobustCrxbQuantizer
    elif args.quantizer == 'SimpleCrxbQuantizer':
        layer_quantizer=SimpleCrxbQuantizer
    elif args.quantizer == 'MinibatchRobustCrxbQuantizer':
        layer_quantizer=MinibatchRobustCrxbQuantizer

    import copy
    model_origin = copy.deepcopy(model)
    model = convert(model_origin, inplace=False, 
            layer_quantizer=layer_quantizer,
            wrap_fc=False, wrapped_modules=wrapped_modules,crxb_cfg=crxb_cfg)
    print("After wrapped\n"+"*"*10+"\n", model)
    calib_model(distributed, args, model, wrapped_modules, calib_loader, calib_dataset, CLASSES)
    dict_bk = copy.deepcopy(model.state_dict())

    f1 = open("{}-sweep_result_acc1.csv".format(model._get_name()), mode="a", newline='')
    csv_f1 = csv.writer(f1)
    csv_f1.writerow(["name", "gamma", "noise_amp", "acc@1", "noisy acc@1", "total error", "quant error", "noise error"])
    result_lists = []
    # layer-by-layer tuning
    for n, (name, module) in enumerate(model.named_modules()):
        if name in wrapped_modules.keys():
            layer_result_list:List[Union[AnyStr, List]] = [name,]
            # scan parameters
            for gamma_t in np.linspace(0, 2, num=11, endpoint=True):
                # reset layer(module) parameters
                module.reset_model()
                print("| ",module.mapped, module.quantizer.calibrated, 
                module.quantizer.w_quantizer.calibrated, 
                module.quantizer.o_quantizer.calibrated,
                module.quantizer.i_quantizer.calibrated," |")
                # set new parameters
                gamma = gamma_t * 1
                module.quantizer.w_quantizer.gamma=gamma
                # calib layer
                acc1 = calib_model(distributed, args, model, wrapped_modules, calib_loader, calib_dataset, CLASSES)
                layer_result_list.append(["gamma:%.2f"%gamma, "acc@1%.3f"%acc1])
                # test layer output error
                # noise_sweep_test("{}".format(gamma), distributed, args, model, calib_loader, calib_dataset, CLASSES, layer_result_list)

                acc_lst = [name]
                step = args.step
                for noise_amp in np.arange(0, args.noise_amp + step, step):
                    amp_result_list = [name]
                    # register hook
                    hook = module.add_calib_hook(False, amp_result_list)
                    # Do mento-carlo simulation with random seeds
                    if args.random_noise:
                        seeds = torch.randint(1000, (args.random_ite,))
                        acc_lst_t = []

                        for rand_seed in seeds:
                            torch.manual_seed(rand_seed)
                            
                            # TODO: fix input of the following layers or remove the previous variation
                            # update_noise(args, model, args.enable_SAF, True, noise_amp, True)
                            # Inject variation
                            module.phy.update_variation(args.enable_noise, noise_amp, args.random_noise, noise_mean=0.)
                            if args.enable_noise:
                                print("variation of std={} is injected into {}".format(noise_amp * 100, name))
                            
                            noisy_acc1 = test_model_acc(distributed, args, model, calib_loader, calib_dataset, CLASSES)
                            acc_lst_t.append(noisy_acc1)
                        amp_result_list.pop(0)
                        data = np.array(amp_result_list)
                        idx = np.argsort(data[:, 0])[len(data)//2]
                        layer_result_list[-1].append(["noise_amp:%.2f"%noise_amp, amp_result_list[idx], "noisy acc@1：%.3f"%acc_lst_t[idx]]) 
                        csv_f1.writerow([name, gamma, noise_amp, acc1, acc_lst_t[idx], *amp_result_list[idx]])
                    else:
                        noisy_acc1 = test_model_acc(distributed, args, model, calib_loader, calib_dataset, CLASSES)
                        acc_lst.append(noisy_acc1)
                    # remove hook
                    hook.remove()
                # recover parameters
                # TODO: only load current layer
                model.load_state_dict(dict_bk)
                # cancel the noise injection
                module.phy.update_variation(False)

            result_lists.append(layer_result_list)
    print(result_lists)
    f1.close()

# N_l*{N_gamma*{N_amp*{[diff, diff1, diff2]}}}


if __name__ == '__main__':
    main()