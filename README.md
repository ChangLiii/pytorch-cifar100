# CS330 Project Learning to Augment

Implementation of CS330 Project Learning to Augment. This codebase is forked from https://github.com/weiaicunzai/pytorch-cifar100, and our implementation is built on top of the original codebase.

## Requirements

This is my experiment eviroument
- python3.6
- pytorch1.6.0+cu101
- tensorboard 2.2.2(optional)


## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100
```

### 2. dataset
I will use cifar100 dataset from torchvision since it's more convenient, but I also
kept the sample code for writing your own dataset module in dataset folder, as an
example for people don't know how to write it.

### 3. run tensorbard(optional)
Install tensorboard
```bash
$ pip install tensorboard
$ mkdir runs
Run tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

### 4. train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train preactresnet18 classification model without learning augmentation module
$ python train.py -net preactresnet18 -gpu -lr 0.1 -augmentation_lr 0.01

# use gpu to train preactresnet18 classification model and learning the aumgentation model that outputs a single value E2E
$ python train.py -net preactresnet18 -gpu -learning_augmentation -augmentation_mode value -lr 0.1 -augmentation_lr 0.01

# use gpu to train preactresnet18 classification model and learning the aumgentation model that outputs a distribution E2E
$ python train.py -net preactresnet18 -gpu -learning_augmentation -augmentation_mode distribution -lr 0.1 -augmentation_lr 0.01

# use gpu to train preactresnet18 classification model and learning the aumgentation model that outputs a single value using 2-step training strategy
$ python train.py -net preactresnet18 -gpu -learning_augmentation -augmentation_mode value -two_step -lr 0.1 -augmentation_lr 0.01 -lr_ratio 1.0

# use gpu to train preactresnet18 classification model and learning the aumgentation model that outputs a distribution using 2-step training strategy
$ python train.py -net preactresnet18 -gpu -learning_augmentation -augmentation_mode distribution -two_step -lr 0.1 -augmentation_lr 0.01 -lr_ratio 1.0
```

sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
diverge during early training phase.
Also you can adjust augmentation model learning rate seperately using ```-augmentation_lr```.
And in 2-step training paradigm, you can use ```-lr_ratio``` to adjust the inner learning rate used in step 1 repsect to the outter learning rate used in step 2.

The supported net args are:
```
squeezenet
mobilenet
mobilenetv2
shufflenet
shufflenetv2
vgg11
vgg13
vgg16
vgg19
densenet121
densenet161
densenet201
googlenet
inceptionv3
inceptionv4
inceptionresnetv2
xception
resnet18
resnet34
resnet50
resnet101
resnet152
preactresnet18
preactresnet34
preactresnet50
preactresnet101
preactresnet152
resnext50
resnext101
resnext152
attention56
attention92
seresnet18
seresnet34
seresnet50
seresnet101
seresnet152
nasnet
wideresnet
stochasticdepth18
stochasticdepth34
stochasticdepth50
stochasticdepth101
```
Normally, the weights file with the best accuracy would be written to the disk with name suffix 'best'(default in checkpoint folder).


### 5. test the model
Test the model using test.py
```bash
$ python test.py -net vgg16 -weights path_to_vgg16_weights_file
```

## Training Details
For training details, please refer to our final project paper. Thanks!

## Results
For detailed experiment results, please refer to our final project paper. Thanks!