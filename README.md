# How to break CIFAR10

Code for the contest of Vision and Cognitive Service - VCS AY 2020-2021

### Install
```bash
# Clone the repo 
$ git clone https://github.com/guglielmocamporese/break_cifar10.git break_cifar10

# Go the in project folder
$ cd break_cifar10
```

You need to have the main packages for training NNS... (PyTorch, PyTorch Lightning, ...)

### Train and Evaluate a model
```
# Run the main
$ python main.py --mode train --model vit

# Test
$ python main.py --mode test --model vit --model_checkpoint 'your/model/checkpoint.ckpt'
```

You can change the model in the `config.py` file. Supported models are `ResNet-18`, `ResNet-34`, `ResNet50`, 'ViT' and 'MLP-Mixer'.

