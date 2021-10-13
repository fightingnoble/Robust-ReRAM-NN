# from sklearn.cluster._kmeans import *
import copy
from typing import Union

import torch
import torch.nn as nn
from sklearn.cluster._kmeans import *

from .quantizer import Quantizer

__all__ = ['MiniBatchKMeansTorch', 'KMeansTorch']


class ClusterQuantizerBase(Quantizer):
    def __init__(self, n_feature=1, n_clusters=8, name='',
                 quant_fun=lambda x: x):
        super(ClusterQuantizerBase, self).__init__()
        self.n_clusters = n_clusters
        self.name = name

        # specify the initial values for loading judgment
        self.register_buffer("labels_", torch.zeros((0, ),dtype=torch.long))
        # specify the initial values for initial judgment
        self.register_buffer("cluster_centers_", torch.zeros(n_clusters, n_feature))
        self.quant_fun = quant_fun

    def forward(self, inputs):
        output = self.quant_func(inputs)
        return output

    def extra_repr(self) -> str:
        return 'name={},cluster={}'.format(self.name, self.n_clusters)

    @staticmethod
    def quant_calib(net,wrapped_modules,calib_loader):
        calib_layers=[]
        n_calibration_steps=1
        for name,module in wrapped_modules.items():
            module.mode='calibration_forward'
            calib_layers.append(name)
            n_calibration_steps=max(n_calibration_steps,module.quantizer.n_calibration_steps)
        print(f"prepare calibration for {calib_layers}\n n_calibration_steps={n_calibration_steps}")
        for step in range(n_calibration_steps):
            print(f"Start calibration step={step+1}")
            for name,module in wrapped_modules.items():
                module.quantizer.calibration_step=step+1
            with torch.no_grad():
                for inp,target in calib_loader:
                    inp=inp.cuda()
                    net(inp)
        for name,module in wrapped_modules.items():
            print(f"{name}: {module.quantizer}")
            module.mode='qat_forward'
        print("calibration finished")


class KMeansTorch(ClusterQuantizerBase):
    def __init__(self, # data_or_size, 
                 n_feature=1, n_clusters=8, name='',
                  q_level_init='uniform', **kwargs):
        super(KMeansTorch, self).__init__(n_feature, n_clusters=n_clusters, name=name)

        self.kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        # if hasattr(data_or_size, '__array__'):
        #     data = data_or_size
        # else:
        #     data = None
        # # if isinstance(data, torch.Tensor):
        # #     data = data.detach().clone().cpu().view(-1, 1).numpy()
        # if isinstance(data, np.ndarray):
        #     data = self.label_.new_tensor(torch.from_numpy(data))
        # self.init_layer_cluster_center(data, n_clusters, q_level_init)
        self.init_layer_cluster_center(None, n_clusters, q_level_init)

    def fit(self, X: torch.Tensor, y=None, sample_weight=None, n_init=None, init=None, tol=None):
        # 210626 data copy optimization
        # data = X.detach().clone().view(-1, 1)
        data = X.view(-1, 1)
        if X.requires_grad:
            data = data.detach()
        data = data.cpu().numpy()
        bak = copy.deepcopy([self.kmeans.n_init, self.kmeans.init, self.kmeans.tol])
        self.kmeans.n_init, self.kmeans.init, self.kmeans.tol = [new if new is not None else old
                                                                 for new, old in zip((n_init, init, tol), bak)]
        self.kmeans.fit(data, y=y, sample_weight=sample_weight)
        # self.labels_.data.copy_(torch.from_numpy(self.kmeans.labels_))
        self.register_buffer("labels_", torch.as_tensor(self.kmeans.labels_,dtype=torch.long))
        self.cluster_centers_.data.copy_(torch.from_numpy(self.kmeans.cluster_centers_))
        self.kmeans.n_init, self.kmeans.init, self.kmeans.tol = bak

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        # 210626 data copy optimization
        # data = X.detach().clone().view(-1, 1)
        data = X.view(-1, 1)
        if X.requires_grad:
            data = data.detach()
        data = data.cpu().numpy()
        return self.kmeans.predict(data, sample_weight)

    def init_layer_cluster_center(self, data, n_clusters, method="uniform"):
        if method == "uniform" or data is None:
            self.cluster_centers_.data.copy_(torch.linspace(-1, 1, steps=n_clusters).view(-1, 1))
            self.kmeans.cluster_centers_ = self.cluster_centers_.data.cpu().numpy()
        else:
            self.fit(data, tol=1e-2)

    def forward(self, inputs):

        # To avoid fault fitness in initial iterations
        # if (self.cluster_centers_.data == 0).all():
        #     # use uniform quantization to avoid further fitness with bad data
        #     self.init_layer_cluster_center(inputs, self.weight_qbit)
        
        if self.calibration and not self.calibrated:
            self.fit(inputs) 
            labels = self.labels_
            weight_quan = self.cluster_centers_[:, 0][labels].view(inputs.shape)
        elif self.training:
            # label should change as weights are updated
            labels = self.predict(inputs)
            weight_quan_temp = self.cluster_centers_[:, 0][labels].view(inputs.shape)
            weight_quan = inputs - inputs.detach() + weight_quan_temp
        else:
            # to avoid load the model without pre-fitness
            # if len(self.labels_.data) == 0:
            #     # self.labels_.data.copy_(torch.from_numpy(self.predict(inputs)).view(-1))
            #     self.register_buffer("labels_", torch.from_numpy(self.predict(inputs)).view(-1))
            assert len(self.labels_.data)
            labels = self.labels_
            weight_quan_temp = self.cluster_centers_[:, 0][labels].view(inputs.shape)
            weight_quan = weight_quan_temp
        return weight_quan


class MiniBatchKMeansTorch(KMeansTorch):
        
    def __init__(self, # batch_size, # data_or_size, 
                 n_feature=1, n_clusters=8, name='',
                 q_level_init='uniform', **kwargs):
        if "batch_size" in kwargs:
            kwargs.pop("batch_size")
        super().__init__(n_feature=n_feature, n_clusters=n_clusters, name=name, 
                         q_level_init=q_level_init, **kwargs)

        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters,**kwargs)
        # if hasattr(data_or_size, '__array__'):
        #     data = data_or_size
        # else:
        #     data = None
        # # if isinstance(data, torch.Tensor):
        # #     data = data.detach().clone().cpu().view(-1, 1).numpy()
        # if isinstance(data, np.ndarray):
        #     data = self.label_.new_tensor(torch.from_numpy(data))
        # self.init_layer_cluster_center(data, n_clusters, q_level_init)
        self.init_layer_cluster_center(None, n_clusters, q_level_init)

    def partial_fit(self, X, y=None, sample_weight=None):
        """Update k means estimate on a single mini-batch X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Coordinates of the data points to cluster. It must be noted that
            X will be copied if it is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).

        Returns
        -------
        self
        """
        # 210626 data copy optimization
        # data = X.detach().clone().view(-1, 1)
        data = X.view(-1, 1)
        if X.requires_grad:
            data = data.detach()
        data = data.cpu().numpy()
        self.kmeans.partial_fit(data, y, sample_weight)
        # self.labels_.data.copy_(torch.from_numpy(self.kmeans.labels_))
        self.register_buffer("labels_", torch.as_tensor(self.kmeans.labels_,dtype=torch.long))
        self.cluster_centers_.data.copy_(torch.from_numpy(self.kmeans.cluster_centers_))


# TODO: Use close package
def insert_robust_quntizer(module:nn.Module, quantizer: Union[KMeansTorch, MiniBatchKMeansTorch]):
    for k, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            n_samples = m.weight.numel()

            n_clusters = 2 ** m.quanizer.w_bit - 1
            batch_factor = 800
            # if q_type == 'robust_batch':
            if isinstance(quantizer, MiniBatchKMeansTorch):
                m.quantizer.w_quantizer = MiniBatchKMeansTorch(n_feature=1,
                                                                n_clusters=n_clusters,
                                                                batch_size=n_clusters * batch_factor
                                                                if n_clusters * batch_factor < int(0.3 * n_samples)
                                                                else int(0.2 * n_samples),
                                                                n_init=1, max_iter=30, random_state=0,
                                                                q_level_init="uniform"
                                                                )
            # elif q_type == 'robust':
            elif isinstance(quantizer, KMeansTorch):
                m.quantizer.w_quantizer = KMeansTorch(n_feature=1,
                                                       n_clusters=n_clusters,
                                                       n_init=1, max_iter=30, random_state=0,
                                                       q_level_init="uniform"
                                                       )




if __name__ == '__main__':
    import numpy as np

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    torch.set_printoptions(3)

    import sklearn
    sklearn.show_versions()

    a = {}
    # vgg = models.vgg11(pretrained=True)
    # if torch.cuda.is_available():
    #     vgg.cuda()
    # a['state_dict'] = vgg.state_dict()
    a = torch.load("plot/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth",
                   map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'))
    num_class = 7
    batch_factor = 800
    train_flg = False
    kmeans_torch_batch = []
    kmeans_torch = []

    kmeans_sklean = []
    kmeans_sklean_batch = []
    for n, v in a['state_dict'].items():
        if "weight" in n:
            n_samples = v.numel()
            if n_samples > 1024:
                print(n_samples)
                # from sklearn
                kmeans_sklean.append(
                    KMeans(n_clusters=num_class, n_init=1, max_iter=30, random_state=0, algorithm="full"))
                kmeans_sklean_batch.append(
                    MiniBatchKMeans(n_clusters=num_class, n_init=1, max_iter=30, random_state=0,  # tol=1e-4,
                                    batch_size=num_class * batch_factor if num_class * 300 < int(
                                        0.3 * n_samples) else int(0.2 * n_samples)))

                # from clusterq
                kmeans_torch_batch_t = MiniBatchKMeansTorch(n_feature=1,
                                                              n_clusters=num_class,
                                                              batch_size=num_class * batch_factor
                                                              if num_class * batch_factor < int(0.3 * n_samples)
                                                              else int(0.2 * n_samples),
                                                              n_init=1, max_iter=30, random_state=0,
                                                              q_level_init="uniform"
                                                              )
                if not train_flg:
                    kmeans_torch_batch_t.eval()
                kmeans_torch_t = KMeansTorch(n_feature=1,
                                               n_clusters=num_class,
                                               n_init=1, max_iter=30, random_state=0,
                                               q_level_init="uniform"
                                               )
                if not train_flg:
                    kmeans_torch_t.eval()

                if torch.cuda.is_available():
                    kmeans_torch_batch_t.cuda()
                    kmeans_torch_t.cuda()
                kmeans_torch.append(kmeans_torch_t)
                kmeans_torch_batch.append(kmeans_torch_batch_t)
    import sys

    sys.path.append("../")
    from utee.misc import time_measurement


    @time_measurement(False, 0, 0)
    def f1(quantizer_list, is_np=False):
        print("start fitting\n")
        ix = 0
        for n, v in a['state_dict'].items():
            if "weight" in n:
                n_samples = v.numel()
                if n_samples > 1024:
                    data_o = v.detach().view(-1, 1)
                    if is_np:
                        data = data_o.cpu().numpy()
                    else:
                        data = data_o.cuda()
                    quantizer_list[ix].fit(data)

                    data_o = v.detach().view(-1, 1)
                    if is_np:
                        datac = data_o.cpu().numpy()
                        t = (datac != data)
                        tt = t if not isinstance(t, np.ndarray) else t.any()
                        # print("data is modified:", tt)
                    else:
                        datac = data_o.cuda()
                        t = (datac != data)
                        tt = t.any().item()
                        # print("data is modified:", tt)

                    if tt:
                        print("max difference:", ((datac - data_o)[t]).max())
                    ix += 1


    # import visdom
    #
    # vis = visdom.Visdom()

    class Visdom():
        def bar(self, *args, **kwargs):
            pass

        def line(self, *args, **kwargs):
            pass


    vis = Visdom()


    def plot(quantizer, name="None", is_np=False):
        print(quantizer.labels_)
        print(quantizer.cluster_centers_)

        # ------------- visdom draw --------------
        # histogram of weight distribution
        qw = quantizer.cluster_centers_[:, 0][quantizer.labels_]  # .view(weight.shape)
        qw_hist = []
        if is_np:
            qw_v = np.unique(qw)
            for v in qw_v:
                qw_hist.append((qw == v).sum())
        else:
            qw_v = qw.unique()
            for v in qw_v:
                qw_hist.append((qw == v).sum().item())
        vis.bar(torch.tensor(qw_hist), qw_v, win=name + " hist",
                opts=dict(title=name + " hist"))
        # vis.histogram(qw, win=name+" hist",
        #               opts=dict(title=name+" hist"))
        # transform function
        x = torch.arange(-1., 1., 0.01)
        print(x.shape)
        if is_np:
            x = x.view(-1, 1).cpu().numpy()
        elif torch.cuda.is_available():
            x = x.view(-1, 1).cuda()
        else:
            x = x.view(-1, 1)

        level1 = quantizer.cluster_centers_[:, 0][quantizer.predict(x)]
        # print(level1.shape, x.shape)

        vis.line(Y=level1, X=x.reshape(-1),
                 win=name,
                 opts=dict(title=name))


    @time_measurement(False, 0, 0)
    def get_q_loss(quantizer_list, is_np=False):
        print("start prediction\n")
        ix = 0
        loss = 0
        for n, v in a['state_dict'].items():
            if "weight" in n:
                n_samples = v.numel()
                if n_samples > 1024:
                    if is_np:
                        data = v.detach().view(-1, 1)
                        data = data.cpu().numpy()
                        q_data = quantizer_list[ix].cluster_centers_[:, 0][quantizer_list[ix].predict(data)].reshape(
                            data.shape)
                    else:
                        data = v
                        q_data = quantizer_list[ix](data).reshape(data.shape)
                    loss += ((q_data - data) ** 2).sum()
                    # print(n)
                    ix += 1
        print(loss)


    print("=======test kmeans_sklean======\n")
    f1(kmeans_sklean, True)
    get_q_loss(kmeans_sklean, True)

    print("=======test kmeans_sklean_batch======\n")
    f1(kmeans_sklean_batch, True)
    get_q_loss(kmeans_sklean_batch, True)

    print("=======test kmeans_torch======\n")
    f1(kmeans_torch)
    get_q_loss(kmeans_torch)
    plot(kmeans_torch[0], 'kmeans_torch')

    print("=======test kmeans_torch_batch======\n")
    f1(kmeans_torch_batch)
    get_q_loss(kmeans_torch_batch)
    plot(kmeans_torch_batch[0], 'kmeans_torch_batch')

    print("=======test uniform======\n")
    from module.quantization.quant_functions import linear_quantize, compute_integral_part

    bits = 3
    print("start\n")
    ix = 0
    q2_loss = 0
    q2_list = []
    for n, v in a['state_dict'].items():
        if "weight" in n:
            n_samples = v.numel()
            if n_samples > 1024:
                w = v.detach()
                sf = bits - 1. - compute_integral_part(w, overflow_rate=0)
                q2 = linear_quantize(w, sf, bits=bits)
                q2_list.append(q2)
                q2_loss += ((q2 - w)**2).sum()
                ix += 1
    print(q2_loss)
    # vis.histogram(q2_list[0].view(-1), win='uniform'+" hist",
    #               opts=dict(title='uniform'+" hist"))
    qw = q2_list[0]
    qw_v = qw.unique()
    qw_hist = []
    for v in qw_v:
        qw_hist.append((qw == v).sum().item())
    vis.bar(torch.tensor(qw_hist), qw_v, win='uniform' + " hist",
            opts=dict(title='uniform' + " hist"))

# 2021/08/31: remove dulplicated code of MiniBatchRobustqTorch and RobustqTorch, 
# 2021/08/31: MiniBatchRobustqTorch inherits functions from RobustqTorch.