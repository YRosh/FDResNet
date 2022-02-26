# FDResNet [[arXiv PDF]](https://arxiv.org/pdf/2109.12556.pdf)

### Settings used

###### CIFAR10 & CIFAR100

- Image size: 32x32
- First Conv block: kernel_size=3, stride=1, padding=1
- Avg pooling

###### Tiny ImageNet and Caltech-256

- Image size: 64x64 and 128x128
- First Conv block: kernel_size=7, stride=2, padding=3
- Adaptive pooling

Code inspired [from](https://github.com/kuangliu/pytorch-cifar)

FDResNet paper available on arxiv: [arXiv Paper](https://arxiv.org/pdf/2109.12556.pdf)

If you use FDResNet code in your research, pleasew cite following work:
S.R. Singh, R.R. Yedla, S.R. Dubey, R. Sanodiya, and W.-T. Chu, "Frequency Disentangled Residual Network", arXiv preprint arXiv:2109.12556, 2021.
