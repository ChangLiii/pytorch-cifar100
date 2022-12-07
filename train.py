# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import init

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

import time

import numpy as np
import torch
import torch.optim as optim
from augmentation_module import Augmentation_Module

def permute_list(list):
    indices = np.random.permutation(len(list))
    return [list[i] for i in indices]


def forward_and_backward(model, data, label, inner_lr=0.04, outer_lr=0.04):
    assert data.shape[0] == label.shape[0], 'data label must be of the same length'
    data_len = data.shape[0]
    data_0 = data[:data_len//2]
    data_1 = data[data_len//2:]
    label_0 = label[:data_len//2]
    label_1 = label[data_len//2:]

    # forward
    model.train()
    augmentation_module.train()

    trainable_params = {}
    for k, v in model.named_parameters():
        if v.requires_grad:
            trainable_params[k] = v

    model_names = trainable_params.keys()
    w = trainable_params.values()

    adjusted_data_0, brightness_factor = adjust_brightness(data_0, augmentation_module, mode='learnable')
    output_0 = model(adjusted_data_0)
    loss = loss_function(output_0, label_0)
    gw = torch.autograd.grad(loss, w, create_graph=True)

    new_trainable_params = {}
    for i, a_name in enumerate(model_names):
        new_trainable_params[a_name] = trainable_params[a_name] - gw[i] * inner_lr
        
    # final L
    model.eval()
    w_modules_names = []
    for m in model.modules():
        for n, p in m.named_parameters(recurse=False):
            if p is not None:
                w_modules_names.append((m, n))
    for (m, n), w in zip(w_modules_names, new_trainable_params.values()):
        setattr(m, n, nn.Parameter(w))
    output_1 = model(data_1)
    model.train()
    new_model_trainable_params = {}
    for k, v in model.named_parameters():
        if v.requires_grad:
            new_model_trainable_params[k] = v
    final_loss = loss_function(output_1, label_1)
    dw = torch.autograd.grad(final_loss, new_model_trainable_params.values(), retain_graph=True)
    trainable_module_params = {}
    for k, v in augmentation_module.named_parameters():
        if v.requires_grad:
            trainable_module_params[k] = v
    dgw = (-a_dw for a_dw in dw)
    augmentation_grad = torch.autograd.grad(
        outputs=gw,
        inputs=trainable_module_params.values(),
        grad_outputs=dgw,
        retain_graph=True
    )
    torch.autograd.backward(trainable_module_params.values(), grad_tensors=augmentation_grad, retain_graph=True)
    torch.autograd.backward(new_model_trainable_params.values(), grad_tensors=dw)
    # final_loss.backward()
    return final_loss, brightness_factor

class AugmentationModule(nn.Module):
    def __init__(self):
        super(AugmentationModule, self).__init__()
        self.learn_brightness_factor = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=2),
            torch.nn.Sigmoid(),
            torch.nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        brightness_factor = self.learn_brightness_factor(x)
        brightness_factor.squeeze_(-1).squeeze_(-1).squeeze_(-1)
        return brightness_factor



def adjust_brightness(images: torch.Tensor, augmentation_module, mode='random', brightness=[0.2, 2]):
    if mode=='random':
        num_sample_per_batch = images.shape[0]
        brightness_factor = torch.empty(num_sample_per_batch).uniform_(brightness[0], brightness[1])
        if args.gpu:
            brightness_factor = brightness_factor.cuda()
        image_upper_bound = 1.0
        adjusted_images = brightness_factor[:,None, None, None] * images.clamp(0, image_upper_bound).to(images.dtype)
    # augmentation_module v1
    # elif mode=='learnable':
    #     brightness_factor = augmentation_module(images)
    #     if brightness is not None:
    #         brightness_factor = brightness_factor.clamp(brightness[0], brightness[1])
        # image_upper_bound = 1.0
        # adjusted_images = brightness_factor[:,None, None, None] * images.clamp(0, image_upper_bound).to(images.dtype)
    
    elif mode=='learnable':
        adjusted_images, brightness_factor = augmentation_module(images)

    return adjusted_images, brightness_factor

def init_weights(net, state):
    init_type, init_param = state.init, state.init_param

    if init_type == 'imagenet_pretrained':
        assert net.__class__.__name__ == 'AlexNet'
        state_dict = torchvision.models.alexnet(pretrained=True).state_dict()
        state_dict['classifier.6.weight'] = torch.zeros_like(net.classifier[6].weight)
        state_dict['classifier.6.bias'] = torch.ones_like(net.classifier[6].bias)
        net.load_state_dict(state_dict)
        del state_dict
        return net

    def init_func(m):
        classname = m.__class__.__name__
        if classname.startswith('Conv') or classname == 'Linear':
            if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias, 0.0)
            if getattr(m, 'weight', None) is not None:
                if init_type == 'normal':
                    init.normal_(m.weight, 0.0, init_param)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=init_param)
                elif init_type == 'xavier_unif':
                    init.xavier_uniform_(m.weight, gain=init_param)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
                elif init_type == 'kaiming_out':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=init_param)
                elif init_type == 'default':
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif 'Norm' in classname:
            if getattr(m, 'weight', None) is not None:
                m.weight.data.fill_(1)
            if getattr(m, 'bias', None) is not None:
                m.bias.data.zero_()

    net.apply(init_func)
    return net


def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        inner_lr = lr_ratio * optimizer.param_groups[0]['lr']
        loss, brightness_factor = forward_and_backward(net, images, labels, inner_lr)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        
        mean_brightness_factor = float(torch.mean(brightness_factor))
        writer.add_scalar('brightness_factor/mean', mean_brightness_factor, n_iter)
        min_brightness_factor = float(torch.min(brightness_factor))
        writer.add_scalar('brightness_factor/min', min_brightness_factor, n_iter)
        max_brightness_factor = float(torch.max(brightness_factor))
        writer.add_scalar('brightness_factor/max', max_brightness_factor, n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}\tAugLR: {:0.6f} mean_brightness_factor: {:0.4f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            optimizer.param_groups[1]['lr'],
            mean_brightness_factor,
            mean_brightness_factor,
            max_brightness_factor,
            min_brightness_factor,
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, model: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-augmentation_lr', type=float, default=0.01, help='initial learning rate for augmentation module')
    parser.add_argument('-lr_ratio', type=float, default=0.1, help='ratio of inner lr to outer lr')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-learning_augmentation', action='store_true', default=False, help='learning augmentation')
    args = parser.parse_args()

    net = get_network(args)

    if args.learning_augmentation:
        augmentation_module = Augmentation_Module() # AugmentationModule()
        if args.gpu: #use_gpu
            augmentation_module = augmentation_module.cuda()

    lr_ratio = args.lr_ratio

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
 
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.learning_augmentation:
        optimizer.add_param_group({'params': augmentation_module.parameters(), 'lr': args.augmentation_lr, 'momentum': 0.9, 'weight_decay': 5e-4})
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
