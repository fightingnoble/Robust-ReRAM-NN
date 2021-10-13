import torch

minimal_num = 1e-12

from .quantizer import AverageLinearSignSymmIntervalQuantizer
from .cluster_q import RobustqTorch

print('Prepare to use decorated function')

quantizer1 = AverageLinearSignSymmIntervalQuantizer(3)
quantizer2 = RobustqTorch(n_feature=1,
                            n_clusters=2 **3 - 1,
                            alpha=0, gamma=0,
                            n_init=1, max_iter=30, random_state=0,
                            q_level_init="uniform"
                            )

for i in range(15):
    a = torch.normal(0,1,size=(100000,))
    a = a/a.abs().max()
    qa1 = quantizer1(a)
    qa2 = quantizer2(a)
    # print(id(quantizer))
    # print(id(quantizer1.sum))

print('Test finished')
print((a-qa1).abs().max())
print((a-qa1).abs().sum())
print((a-qa2).abs().max())
print((a-qa2).abs().sum())
print((qa1-qa2).abs().sum())
