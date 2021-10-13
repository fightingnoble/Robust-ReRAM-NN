python mm_cls.py  configs/vgg/vgg11_b32x8_imagenet.py  checkpoints/vgg11_batch256_imagenet_20210208-4271cd6c.pth --metrics=accuracy --metric-options=topk=1 --w_bits 8 --quantizer SimpleCrxbQuantizer
python mm_cls.py  configs/vgg/vgg11_b32x8_imagenet.py  checkpoints/vgg11_batch256_imagenet_20210208-4271cd6c.pth --metrics=accuracy --metric-options=topk=1 --w_bits 4 --quantizer SimpleCrxbQuantizer

python mm_cls.py  configs/vgg/vgg11_b32x8_imagenet.py  checkpoints/vgg11_batch256_imagenet_20210208-4271cd6c.pth --metrics=accuracy --metric-options=topk=1 --w_bits 8 --quantizer RobustCrxbQuantizer
python mm_cls.py  configs/vgg/vgg11_b32x8_imagenet.py  checkpoints/vgg11_batch256_imagenet_20210208-4271cd6c.pth --metrics=accuracy --metric-options=topk=1 --w_bits 4 --quantizer RobustCrxbQuantizer

python mm_cls.py  configs/vgg/vgg11_b32x8_imagenet.py  checkpoints/vgg11_batch256_imagenet_20210208-4271cd6c.pth --metrics=accuracy --metric-options=topk=1 --w_bits 8 --quantizer MinibatchRobustCrxbQuantizer
python mm_cls.py  configs/vgg/vgg11_b32x8_imagenet.py  checkpoints/vgg11_batch256_imagenet_20210208-4271cd6c.pth --metrics=accuracy --metric-options=topk=1 --w_bits 4 --quantizer MinibatchRobustCrxbQuantizer

python mm_cls.py  configs/lenet/lenet5_mnist.py  checkpoints/lenet/epoch_5.pth  --metrics=accuracy --metric-options=topk=1 --w_bits 8 --quantizer SimpleCrxbQuantizer
python mm_cls.py  configs/lenet/lenet5_mnist.py  checkpoints/lenet/epoch_5.pth  --metrics=accuracy --metric-options=topk=1 --w_bits 4 --quantizer SimpleCrxbQuantizer

python mm_cls.py  configs/lenet/lenet5_mnist.py  checkpoints/lenet/epoch_5.pth  --metrics=accuracy --metric-options=topk=1 --w_bits 8 --quantizer RobustCrxbQuantizer
python mm_cls.py  configs/lenet/lenet5_mnist.py  checkpoints/lenet/epoch_5.pth  --metrics=accuracy --metric-options=topk=1 --w_bits 4 --quantizer RobustCrxbQuantizer

python mm_cls.py  configs/lenet/lenet5_mnist.py  checkpoints/lenet/epoch_5.pth  --metrics=accuracy --metric-options=topk=1 --w_bits 8 --quantizer MinibatchRobustCrxbQuantizer
python mm_cls.py  configs/lenet/lenet5_mnist.py  checkpoints/lenet/epoch_5.pth  --metrics=accuracy --metric-options=topk=1 --w_bits 4 --quantizer MinibatchRobustCrxbQuantizer
