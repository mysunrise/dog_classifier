from __future__ import print_function
from __future__ import division

import argparse
import time
import os
import shutil

from models import *
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152',
                    help='model architecture: vgg19, resnet152')
parser.add_argument('--num_classes', default=100, type=int, metavar='N',
                    help='number of total class')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_freq', '-s', default=10, type=int,
                    metavar='N', help='save frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='.', type=str, metavar='PATH',
                    help='path to save the models')

best_prec1 = 0


def main():
    global best_prec1, args, use_gpu
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # load data
    traindir = './data/new_train'
    valdir = './data/new_test'
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer
        ])),
        batch_size=32, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer
        ])),
        batch_size=16, shuffle=False)

    # create model
    if args.arch == 'vgg19':
        vgg19 = models.vgg19(pretrained=True)
        model = Vgg19Ft(vgg19, args.num_classes)
        ignored_params = list(map(id, list(model.classifier.children())[6].parameters()))
        base_params = filter(lambda x: id(x) not in ignored_params, model.parameters())

        optimizer = optim.SGD([{'params': base_params},
                               {'params': list(model.classifier.children())[6].parameters(), 'lr': args.lr}],
                              args.lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        if args.arch == 'resnet152':
            model = models.resnet152(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, args.num_classes)
            ignored_params = list(map(id, model.fc.parameters()))
            base_params = filter(lambda x: id(x) not in ignored_params, model.parameters())
            optimizer = optim.SGD([{'params': base_params},
                                   {'params': model.fc.parameters(), 'lr': args.lr}],
                                  args.lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            print('please choose reasonable model architeture')
            return

    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        model.cuda()
        criterion.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch))
        else:
            print("no checkpoint found at '{}'".format(args.resume))

    for epoch in xrange(args.start_epoch + 1, args.epochs + 1):
        train_model(model, train_loader, criterion, optimizer, exp_lr_scheduler, epoch)
        prec1 = test_model(model, val_loader, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if epoch % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
            }, epoch, False)
        if is_best:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
            }, epoch, is_best)
            print('best prec1: {}'.format(best_prec1))


'''
def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=40):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('learning rate is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
'''


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=30):
    i = 0
    for param_group in optimizer.param_groups:
        if epoch % lr_decay_epoch == 0:
            param_group['lr'] = param_group['lr'] * 0.1
            print('learning rate for param_group {} is set to {}'.format(i, param_group['lr']))
        i += 1
    return optimizer


def train_model(model, train_loader, criterion, optimizer, lr_scheduler, epoch):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    since = time.time()
    optimizer = lr_scheduler(optimizer, epoch)

    for idx, (input, target) in enumerate(train_loader):
        if use_gpu:
            input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target)

        output = model(input)
        loss = criterion(output, target)
        _, prec1 = torch.max(output.data, 1)
        nums_correct = torch.sum(prec1 == target.data)
        losses.update(loss.data[0], input.size(0))
        top1.update(nums_correct / target.size(0), target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - since)
        since = time.time()

        if idx % args.print_freq == 0:
            print('Epoch: [{0}] [{1}/{2}]\t'
                  'Time {batch_time.val: .3f} {batch_time.avg: .3f}\t'
                  'Loss {loss.val: .4f} {loss.avg: .4f}\t'
                  'top1 {top1.val: .4f} {top1.avg: .4f}\t'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   loss=losses, top1=top1))


def test_model(model, val_loader, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    since = time.time()

    for idx, (input, target) in enumerate(val_loader):
        if use_gpu:
            input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target)

        output = model(input)
        loss = criterion(output, target)
        _, prec1 = torch.max(output.data, 1)
        nums_correct = torch.sum(prec1 == target.data)
        losses.update(loss.data[0], input.size(0))
        top1.update(nums_correct / target.size(0), target.size(0))
        batch_time.update(time.time() - since)
        since = time.time()

        if idx % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val: .3f} {batch_time.avg: .3f}\t'
                  'Loss {loss.val: .4f} {loss.avg: .4f}\t'
                  'top1 {top1.val: .4f} {top1.avg: .4f}\t'.format(
                   idx, len(val_loader), batch_time=batch_time,
                   loss=losses, top1=top1))

    print('Prec@1: {top1.avg: .4f}'.format(top1=top1))

    return top1.avg


class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.val = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, epoch, is_best):
    if is_best:
        torch.save(state, os.path.join(args.save_dir, 'model_best.pth'))
        return
    model_out_path = os.path.join(args.save_dir, 'model_epoch_{}.pth'.format(epoch))
    torch.save(state, model_out_path)
    print('checkpoint saved to {}'.format(model_out_path))


if __name__ == '__main__':
    main()

