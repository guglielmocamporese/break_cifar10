# How to Break the CIFAR-10

Code for the submission of the contest of *VCS AY 2020-2021*, the Vision and Cognitive Service class, University of Padova, Italy.

### Results on CIFAR-10
|   | ResNet18 [[1](https://arxiv.org/pdf/1512.03385.pdf)] | ResNet34 [[1](https://arxiv.org/pdf/1512.03385.pdf)] | ResNet50 [[1](https://arxiv.org/pdf/1512.03385.pdf)] | ResNet Ens [[1](https://arxiv.org/pdf/1512.03385.pdf)] | MLP-Mixer [[2](https://arxiv.org/pdf/2105.01601.pdf)] | ViT-S/16 [[3](https://arxiv.org/pdf/2010.11929.pdf)] |  ViT-B/16 [[3](https://arxiv.org/pdf/2010.11929.pdf)] | 
| - | - | - | - | - | - | - | - |
| **Accuracy** | 95.01 % | 96.92 % | 95.46 % | 97.53 % | 94.67 % | 96.05 % | **98.67 %** |


### Install
```bash
# Clone the repo 
$ git clone https://github.com/guglielmocamporese/break_cifar10.git break_cifar10

# Go the in project folder
$ cd break_cifar10

# Install conda env with all the needed packages
$ conda env create -f environment.yml

# Activate the conda env
$ conda activate torch
```

### Train and Evaluate a model
```
# Run the main
$ python main.py --mode train --model vit

# Test
$ python main.py --mode test --model vit --model_checkpoint 'your/model/checkpoint.ckpt'
```

You can change the model in the `config.py` file. Supported models are `ResNet-18`, `ResNet-34`, `ResNet50`, `ViT` and `MLP-Mixer`.
