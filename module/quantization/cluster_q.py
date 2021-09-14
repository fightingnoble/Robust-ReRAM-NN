# from sklearn.cluster._kmeans import *
import copy
from typing import Union

import torch
import torch.nn as nn
from sklearn.cluster._robustq import *

from .quantizer import Quantizer

__all__ = ['MiniBatchRobustqTorch', 'RobustqTorch']


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
            module.mode='quant_forward'
        print("calibration finished")


class RobustqTorch(ClusterQuantizerBase):
    def __init__(self, # data_or_size, 
                 n_feature=1, n_clusters=8, name='',
                 alpha=0.1, gamma=1.0, q_level_init='uniform', **kwargs):
        super(RobustqTorch, self).__init__(n_feature, n_clusters=n_clusters, name=name)

        self.alpha = alpha
        self.gamma = gamma
        self.kmeans = RobustQ(n_clusters=n_clusters, **kwargs)
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
        self.kmeans.fit(data, y=y, sample_weight=sample_weight, var_std=self.alpha, var_weight=self.gamma)
        # self.labels_.data.copy_(torch.from_numpy(self.kmeans.labels_))
        self.register_buffer("labels_", torch.from_numpy(self.kmeans.labels_))
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
        return self.kmeans.predict(data, sample_weight, var_std=self.alpha, var_weight=self.gamma)

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
        
        if self.training or self.calibration:
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

    def extra_repr(self) -> str:
        return super(RobustqTorch, self).extra_repr() + " gamma:{}, alpha:{} )".format(self.gamma, self.alpha)


class MiniBatchRobustqTorch(RobustqTorch):
        
    def __init__(self, # data_or_size, 
                 n_feature=1, n_clusters=8, name='',
                 alpha=0.1, gamma=1.0, q_level_init='uniform', **kwargs):
        super().__init__(n_feature=n_feature, n_clusters=n_clusters, name=name, 
                         alpha=alpha, gamma=gamma, q_level_init=q_level_init, **kwargs)

        self.kmeans = MiniBatchRobustQ(n_clusters=n_clusters, **kwargs)
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
        self.kmeans.partial_fit(data, y, sample_weight, var_std=self.alpha, var_weight=self.gamma)
        # self.labels_.data.copy_(torch.from_numpy(self.kmeans.labels_))
        self.register_buffer("labels_", torch.from_numpy(self.kmeans.labels_))
        self.cluster_centers_.data.copy_(torch.from_numpy(self.kmeans.cluster_centers_))

    def extra_repr(self) -> str:
        return super(MiniBatchRobustqTorch, self).extra_repr() + " gamma:{}, alpha:{} )".format(self.gamma, self.alpha)


# TODO: Use close package
def insert_robust_quntizer(module:nn.Module, quantizer: Union[RobustqTorch, MiniBatchRobustqTorch], alpha, gamma):
    for k, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            n_samples = m.weight.numel()

            n_clusters = 2 ** m.quanizer.w_bit - 1
            batch_factor = 800
            # if q_type == 'robust_batch':
            if isinstance(quantizer, MiniBatchRobustqTorch):
                m.quantizer.w_quantizer = MiniBatchRobustqTorch(n_feature=1,
                                                                n_clusters=n_clusters,
                                                                alpha=alpha, gamma=gamma,
                                                                batch_size=n_clusters * batch_factor
                                                                if n_clusters * batch_factor < int(0.3 * n_samples)
                                                                else int(0.2 * n_samples),
                                                                n_init=1, max_iter=30, random_state=0,
                                                                q_level_init="uniform"
                                                                )
            # elif q_type == 'robust':
            elif isinstance(quantizer, RobustqTorch):
                m.quantizer.w_quantizer = RobustqTorch(n_feature=1,
                                                       n_clusters=n_clusters,
                                                       alpha=alpha, gamma=gamma,
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
    a = torch.load("/root/cifar10_vgg8_SGD_StepLR_model_best.pth.tar",
                   map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'))
    num_class = 7
    batch_factor = 800
    gamma = 0.
    train_flg = False
    robustq_torch_batch = []
    robustq_sklean_batch = []

    robustq_torch = []
    robustq_sklean = []

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

                # from Robustq
                robustq_sklean.append(
                    RobustQ(n_clusters=num_class, n_init=1, max_iter=30, random_state=0, algorithm="full"))
                robustq_sklean_batch.append(MiniBatchRobustQ(n_clusters=num_class,
                                                             n_init=1, max_iter=30, random_state=0,  # tol=1e-4,
                                                             batch_size=num_class * batch_factor
                                                             if num_class * batch_factor < int(0.3 * n_samples)
                                                             else int(0.2 * n_samples)))
                # from clusterq
                robustq_torch_batch_t = MiniBatchRobustqTorch(data_or_size=v, n_feature=1,
                                                              n_clusters=num_class,
                                                              alpha=0.12, gamma=gamma,
                                                              batch_size=num_class * batch_factor
                                                              if num_class * batch_factor < int(0.3 * n_samples)
                                                              else int(0.2 * n_samples),
                                                              n_init=1, max_iter=30, random_state=0,
                                                              q_level_init="uniform"
                                                              )
                if not train_flg:
                    robustq_torch_batch_t.eval()
                robustq_torch_t = RobustqTorch(data_or_size=v, n_feature=1,
                                               n_clusters=num_class,
                                               alpha=0.12, gamma=gamma,
                                               n_init=1, max_iter=30, random_state=0,
                                               q_level_init="uniform"
                                               )
                if not train_flg:
                    robustq_torch_t.eval()

                if torch.cuda.is_available():
                    robustq_torch_batch_t.cuda()
                    robustq_torch_t.cuda()
                robustq_torch.append(robustq_torch_t)
                robustq_torch_batch.append(robustq_torch_batch_t)
    import sys

    sys.path.append("../")
    from utee.misc import time_measurement


    @time_measurement(False, 0, 0)
    def f1(quantizer_list, is_np=False):
        print("start\n")
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
                        print("data is modified:", tt)
                    else:
                        datac = data_o.cuda()
                        t = (datac != data)
                        tt = t.any().item()
                        print("data is modified:", tt)

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
                opts=dict(title=name + " hist" + ' gamma={}'.format(gamma)))
        # vis.histogram(qw, win=name+" hist",
        #               opts=dict(title=name+" hist"+' gamma={}'.format(gamma)))
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
    # ix = 0
    # loss = 0
    # for n, v in a['state_dict'].items():
    #     if "weight" in n:
    #         n_samples = v.numel()
    #         if n_samples > 1024:
    #             data = v.detach().clone().view(-1, 1)
    #             data = data.cpu().numpy()
    #             q_data = kmeans_sklean[ix].cluster_centers_[:, 0][kmeans_sklean[ix].predict(data)].reshape(
    #                 data.shape)
    #             loss += ((q_data - data) ** 2).sum()
    #             # print(n)
    #             ix += 1
    # print(loss)

    print("=======test kmeans_sklean_batch======\n")
    f1(kmeans_sklean_batch, True)
    get_q_loss(kmeans_sklean_batch, True)
    # ix = 0
    # loss = 0
    # for n, v in a['state_dict'].items():
    #     if "weight" in n:
    #         n_samples = v.numel()
    #         if n_samples > 1024:
    #             data = v.detach().clone().view(-1, 1)
    #             data = data.cpu().numpy()
    #             q_data = kmeans_sklean_batch[ix].cluster_centers_[:, 0][kmeans_sklean_batch[ix].predict(data)].reshape(
    #                 data.shape)
    #             loss += ((q_data - data) ** 2).sum()
    #             # print(n)
    #             ix += 1
    # print(loss)

    print("=======test robustq_sklean======\n")
    f1(robustq_sklean, True)
    get_q_loss(robustq_sklean, True)
    # ix = 0
    # loss = 0
    # for n, v in a['state_dict'].items():
    #     if "weight" in n:
    #         n_samples = v.numel()
    #         if n_samples > 1024:
    #             data = v.detach().clone().view(-1, 1)
    #             data = data.cpu().numpy()
    #             q_data = robustq_sklean[ix].cluster_centers_[:, 0][robustq_sklean[ix].predict(data)].reshape(
    #                 data.shape)
    #             loss += ((q_data - data) ** 2).sum()
    #             # print(n)
    #             ix += 1
    # print(loss)
    plot(robustq_sklean[0], 'robustq_sklean', True)

    print("=======test robustq_sklean_batch======\n")
    f1(robustq_sklean_batch, True)
    get_q_loss(robustq_sklean_batch, True)
    # ix = 0
    # loss = 0
    # for n, v in a['state_dict'].items():
    #     if "weight" in n:
    #         n_samples = v.numel()
    #         if n_samples > 1024:
    #             data = v.detach().clone().view(-1, 1)
    #             data = data.cpu().numpy()
    #             q_data = robustq_sklean_batch[ix].cluster_centers_[:, 0][robustq_sklean_batch[ix].predict(data)].reshape(
    #                 data.shape)
    #             loss += ((q_data - data) ** 2).sum()
    #             # print(n)
    #             ix += 1
    # print(loss)
    plot(robustq_sklean_batch[0], 'robustq_sklean_batch', True)

    print("=======test robustq_torch======\n")
    f1(robustq_torch)
    get_q_loss(robustq_torch)
    # ix = 0
    # loss = 0
    # for n, v in a['state_dict'].items():
    #     if "weight" in n:
    #         n_samples = v.numel()
    #         if n_samples > 1024:
    #             data = v
    #             q_data = robustq_torch[ix].cluster_centers_[:, 0][robustq_torch[ix].predict(data)].reshape(
    #                 data.shape)
    #             loss += ((q_data - data) ** 2).sum()
    #             # print(n)
    #             ix += 1
    # print(loss)
    plot(robustq_torch[0], 'robustq_torch')

    print("=======test robustq_torch_batch======\n")
    f1(robustq_torch_batch)
    get_q_loss(robustq_torch_batch)
    # ix = 0
    # loss = 0
    # for n, v in a['state_dict'].items():
    #     if "weight" in n:
    #         n_samples = v.numel()
    #         if n_samples > 1024:
    #             data = v
    #             q_data = robustq_torch_batch[ix].cluster_centers_[:, 0][robustq_torch_batch[ix].predict(data)].reshape(
    #                 data.shape)
    #             loss += ((q_data - data) ** 2).sum()
    #             # print(n)
    #             ix += 1
    # print(loss)
    plot(robustq_torch_batch[0], 'robustq_torch_batch')

    # print("======= cudalib ======\n")
    # from libKMCUDA import kmeans_cuda
    # clq_temp = []
    # import time
    # t_s = time.monotonic()
    # for n, v in a['state_dict'].items():
    #     if "weight" in n:
    #         n_samples = v.numel()
    #         if n_samples > 1024:
    #             data = v.detach().clone().view(-1, 1)
    #             samples = data.cpu().numpy()
    #             centroids, assignments = kmeans_cuda(samples, num_class, )
    #             clq_temp.append([centroids, assignments])
    # t_e = time.monotonic()
    # s, ms = divmod((t_e - t_s) * 1000, 1000)
    # m, s = divmod(s, 60)
    # h, m = divmod(m, 60)
    # print("%d:%02d:%02d:%03d" % (h, m, s, ms))
    #
    # t_s = time.monotonic()
    # ix = 0
    # loss=0
    # for n, v in a['state_dict'].items():
    #     if "weight" in n:
    #         n_samples = v.numel()
    #         if n_samples > 1024:
    #             data = v.detach().clone().view(-1, 1)
    #             data = data.cpu().numpy()
    #             centroids, assignments = clq_temp[ix]
    #             q_data = centroids[:, 0][assignments].reshape(data.shape)
    #             loss += ((q_data - data) ** 2).sum()
    #             ix +=1
    # t_e = time.monotonic()
    # s, ms = divmod((t_e - t_s) * 1000, 1000)
    # m, s = divmod(s, 60)
    # h, m = divmod(m, 60)
    # print("%d:%02d:%02d:%03d" % (h, m, s, ms))
    # print(loss)

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
                q2_loss += (q2 - w).norm()
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