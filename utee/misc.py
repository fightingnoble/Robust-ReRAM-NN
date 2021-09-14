# import cv2
import copy
import hashlib
import os
import pickle as pkl
import shutil
import time

import numpy as np


class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        self.init('/tmp', 'tmp.log')
        self._logger.info(str_info)


logger = Logger()

print = logger.info


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


def auto_select_gpu(mem_bound=500, utility_bound=0, gpus=(0, 1, 2, 3, 4, 5, 6, 7), num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVCIES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5):  # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert (len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

        if len(ideal_gpus) < num_gpu:
            print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')

    print("Setting GPU: {}".format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus


def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))


def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        arch_n = model.__class__.__name__
        if arch_n.startswith('alexnet') or arch_n.startswith('vgg'):
            model = copy.deepcopy(model)
            model.features = model.features.module
        else:
            model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old models {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving models to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))


# def load_lmdb(lmdb_file, n_records=None):
#     import lmdb
#     import numpy as np
#     lmdb_file = expand_user(lmdb_file)
#     if os.path.exists(lmdb_file):
#         data = []
#         env = lmdb.open(lmdb_file, readonly=True, max_readers=512)
#         with env.begin() as txn:
#             cursor = txn.cursor()
#             begin_st = time.time()
#             print("Loading lmdb file {} into memory".format(lmdb_file))
#             for key, value in cursor:
#                 _, target, _ = key.decode('ascii').split(':')
#                 target = int(target)
#                 img = cv2.imdecode(np.fromstring(value, np.uint8), cv2.IMREAD_COLOR)
#                 data.append((img, target))
#                 if n_records is not None and len(data) >= n_records:
#                     break
#         env.close()
#         print("=> Done ({:.4f} s)".format(time.time() - begin_st))
#         return data
#     else:
#         print("Not found lmdb file".format(lmdb_file))
#
# def str2img(str_b):
#     return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_COLOR)
#
# def img2str(img):
#     return cv2.imencode('.jpg', img)[1].tostring()

def md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()


def eval_model(model, ds, n_sample=None, ngpu=1, is_imagenet=False):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        def forward(self, input):
            input.data.div_(255.)
            input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
            input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
            input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
            return self.model(input)

    correct1, correct5 = 0, 0
    n_passed = 0
    if is_imagenet:
        model = ModelWrapper(model)
    model = model.eval()
    model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda()

    n_sample = len(ds) if n_sample is None else n_sample
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        n_passed += len(data)
        data = Variable(torch.FloatTensor(data)).cuda()
        indx_target = torch.LongTensor(target)
        output = model(data)
        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum()

        if idx >= n_sample - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5


def load_state_dict(model, model_urls, model_root):
    from torch.utils import model_zoo
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict()  # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = model_zoo.load_url(model_urls, model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            print(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    missing = set(own_state.keys()) - set(state_dict.keys())
    no_use = set(state_dict.keys()) - set(own_state.keys())
    if len(no_use) > 0:
        raise KeyError('some keys are not used: "{}"'.format(no_use))


# -------------2020/04/16------------
# add universal load parameter module

def load_state_dict_universe(model, checkpoint):
    from collections import OrderedDict
    import warnings
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['state_dict']
    error_msgs = []
    # print(model_dict.keys())
    # print(pretrained_dict.keys())

    # fit models saved in parallel models
    # -------------2020/09/11------------
    # We can identify the type of variable to determine the operation
    # if list(pretrained_dict.keys())[0].find("module.") != -1:
    #     for k in list(pretrained_dict.keys()):
    #         # 20210315:add:amend comment
    #         # use k[7:] to remove "module."(whose length is 7) prefix of k in the pretrained_dict
    #         pretrained_dict[k[7:]] = pretrained_dict.pop(k)
    for k in list(pretrained_dict.keys()):
        # 20210315:add:amend comment
        # use k[7:] to remove "module."(whose length is 7) prefix of k in the pretrained_dict
        i = k.find("module.")
        if i != -1:
            pretrained_dict[k[:i] + k[i + 7:]] = pretrained_dict.pop(k)
    # -------------2020/09/11------------

    new_dict = OrderedDict()
    for k, v in model_dict.items():
        if k in pretrained_dict:
            new_dict[k] = model_dict[k].copy_(pretrained_dict[k].data)
            # print(k, ' ')
        # -------------2020/09/11------------
        # We can identify the type of variable to determine the operation
        # elif 'module.' + k in pretrained_dict:
        #     new_dict[k] = model_dict[k].copy_(pretrained_dict['module.' + k].data)
        #     pretrained_dict.pop('module.' + k)
        #     # print(k, ' ')
        # -------------2020/09/11------------

        else:
            pass
            # new_dict[k] = v
            # print(k, '!!')

    missing = set(model_dict.keys()) - set(new_dict.keys())
    no_use = set(pretrained_dict.keys()) - set(new_dict.keys())
    if len(no_use) > 0:
        error_msgs.insert(
            0, 'Unexpected key(s) in state_dict: {}. '.format(
                ', '.join('"{}"'.format(k) for k in list(no_use))))

    if len(missing) > 0:
        error_msgs.insert(
            0, 'Missing key(s) in state_dict: {}. '.format(
                ', '.join('"{}"'.format(k) for k in missing)))

    if len(error_msgs) > 0:
        warnings.warn('Error(s) in loading state_dict for {}:\n\t{}'.format(
            model.__class__.__name__, "\n\t".join(error_msgs)))

    # if len(missing): warnings.warn("some keys are not used:)
    # for k in list(missing):
    #     new_dict[k] = model_dict[k]
    # model_dict.update(new_dict)

    try:
        load_state_dict_(model, model_dict)
    except Exception as result:
        print(result)


from typing import Union, Dict
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
import warnings


def load_state_dict_(model, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                     strict: bool = True):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    Arguments:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
    """
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model)
    load = None  # break load->load reference cycle

    if len(unexpected_keys) > 0:
        error_msgs.insert(
            0, 'Unexpected key(s) in state_dict: {}. '.format(
                ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        error_msgs.insert(
            0, 'Missing key(s) in state_dict: {}. '.format(
                ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
        if strict:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        else:
            warnings.warn('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))

    return _IncompatibleKeys(missing_keys, unexpected_keys)


import torch


class data_prefetcher():
    def __init__(self, loader, mean, std, device):
        self.loader = loader.__iter__()
        self.stream = torch.cuda.Stream(device=device)
        mean = torch.as_tensor(mean, device=device)
        std = torch.as_tensor(std, device=device)
        if mean.ndim == 1:
            mean = mean[:, None, None]
        if std.ndim == 1:
            std = std[:, None, None]
        self.mean = mean * 255
        self.std = std * 255
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    c = imgs[0].size[2] if len(imgs[0].size) == 3 else 1
    tensor = torch.zeros((len(imgs), c, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.array(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i].data = torch.from_numpy(nump_array)

    return tensor, targets


def preload(loader):
    try:
        next_data = next(loader)
    except StopIteration:
        next_data = None
    return next_data


import torch.nn as nn


# integrate the update in to the module
# define module apply function
@torch.no_grad()
def update_quantizer(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if hasattr(m, 'layer_type'):
            if m.q_type in ['robust', 'robust_batch', 'clq_new']:
                m.update()


# 20201008-modified: adapt new NN_array layer
def update_noise(args, model, enable_SAF, new_SAF_mask, noise_amp, random_noise, p_SA0=0.1, p_SA1=0.1):
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            n = n + 1
            if args.array_flag:
                if args.enable_noise:
                    m.phy.update_variation(noise_amp, random_noise)
                if args.enable_SAF:
                    m.phy.update_SAF(enable_SAF, p_SA0, p_SA1, new_SAF_mask, enable_rand=False)
            else:
                m.noise_amp = noise_amp
                m.random_noise = random_noise

    if args.enable_noise:
        print("{}% noise is injected into {} layers".format(noise_amp * 100, n))
    if args.enable_SAF:
        print("SAF-{}SA0-{}SA1 is injected into {} layers".format(p_SA0, p_SA1, n))


from functools import wraps


def func_dist(distributed, rank, ngpus_per_node):
    def fun_exe_on_specified_gpu(fun):
        @wraps(fun)
        def wrapped_function(*args, **kwargs):
            if not distributed or (distributed and rank % ngpus_per_node == 0):
                fun(*args, **kwargs)
            pass

        return wrapped_function

    return fun_exe_on_specified_gpu


def time_measurement(distributed, rank, ngpus_per_node):
    def fun_exe_on_specified_gpu(fun):
        @wraps(fun)
        def wrapped_function(*args, **kwargs):
            t_s = time.monotonic()
            output = fun(*args, **kwargs)
            t_e = time.monotonic()
            s, ms = divmod((t_e - t_s) * 1000, 1000)
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            if not distributed or (distributed and rank % ngpus_per_node == 0):
                print("%d:%02d:%02d:%03d" % (h, m, s, ms))
            return output

        return wrapped_function

    return fun_exe_on_specified_gpu
