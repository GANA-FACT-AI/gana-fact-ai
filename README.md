# Fairness, Accountability, Confidentiality and Transparency in AI
This repository holds the code that is used to reproduce the paper by Xiang et al. (2020) [[1]](#1).

## Group members
* Alko Knijff (Student-id: 13413627)
* Noud Corten (Student-id: 11349948)
* Arsen Sheverdin (Student-id: 13198904)
* Georg Lange (Student-id: 13405373)

## Prerequisites
* Modules specified in environment.yml (see 'Installing environment')
* For training, an NVIDIA GPU is strongly recommended. CPU is also supported but significantly decreases training speed.

## Datasets
The following datasets will be downloaded automatically when running the code (see 'Reproducing Experiments'), which requires 1.3GB of available disk space:
* CIFAR-10 (163 MB) - https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
* CIFAR-100 (161 MB) - https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
* CelebA (1GB) - https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8

For testing the VGG-16 network the following dataset needs to be downloaded manually, which requires 1.1GB of available disk space:
* CUB-200 (1.1GB) - http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

## Pretrained Models
Pretrained models for our experiments are available via this Google Drive folder: https://drive.google.com/drive/folders/1kHFsf91qUI1Ob9jz7YrxuHeqVjIJgDiO?usp=sharing

## Installing Environment
To install the environment in Anaconda use the following command:
```console
conda env create -f environment.yml
```
To then activate this environment use:
```console
conda activate factai
```

## Training models
For training a model use the following command
```console
python train.py --model [MODEL] --dataset [DATASET] --progress_bar
```

For training the angle predictor use the following command
```console
python train_discriminator.py --model [MODEL] --dataset [DATASET] --progress_bar
```

For training the adversary use the following command
```console
python train_adversary.py --model [MODEL] --dataset [DATASET] --attack_model [ATTACK] --progress_bar
```

## Testing models / Reproducing results
For testing the models you can use the provided notebook and use the instructions given there.

## References
<a id="1">[1]</a> 
Liyao Xiang, Haotian Ma, Hao Zhang, Yifan Zhang, Jie Ren, and Quanshi Zhang (2020), 
Interpretable complex-valued neural networks for privacy protection. 

<a id="2">[2]</a>
Sébastien M. P. (2021) wavefrontshaping/complexPytorch, 
https://github.com/wavefrontshaping/complexPyTorch

<a id="3">[3]</a>
akamaster (2019) pytorch_resnet_cifar10/resnet.py,
https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py


