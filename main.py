from __future__ import print_function
from __future__ import division

import time
import shutil
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

traindir = './data/train'
valdir = './data/val'
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ])),
    batch_size=64, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalizer
    ])),
    batch_size=64, shuffle=False)

use_gpu = torch.cuda.is_available()
print_frequency = 20

original_model = models.vgg19(pretrained=True)
'''
for param in original_model.parameters():
    print(param)

for name, param in original_model.named_parameters():
    print(name, param.size())

for name, module in original_model.named_children():
    print(name, module)
for name, param in original_model.classifier.named_parameters():
    print(name, param.data)

for name, module in original_model.named_modules():
    print(name, module)
'''


class MyVgg19(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(MyVgg19, self).__init__()
        self.features = pretrained_model.features
        mod = list(pretrained_model.classifier.children())[:-1]
        mod.append(nn.Linear(4096, num_classes))
        self.classifier = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# print(*list(original_model.classifier.children())[:-1])
# print(list(original_model.classifier.children())[:-1])

num_classes = 100
model = MyVgg19(original_model, num_classes)
criterion = nn.CrossEntropyLoss()
if use_gpu:
    model.cuda()
    criterion.cuda()
# print(list(list(model.classifier.children())[0].parameters()))
# print(list(list(model.classifier.children())[6].named_parameters()))
# print(list(original_model.classifier.children())[6])

'''
for name, param in model.named_parameters():
    if name not in list(list(model.classifier.children())[6].named_parameters()):
        print(name)
        param.requires_grad = False
'''
'''
for name, param in list(model.classifier.children())[6].named_parameters():
    print(name, param.size())
for name, param in model.named_parameters():
    print(name)
for name, param in model.classifier.named_parameters():
    print(name, param.size())
'''

ignored_params = list(map(id, list(model.classifier.children())[6].parameters()))
base_params = filter(lambda x: id(x) not in ignored_params, model.parameters())
for param in base_params:
    param.requires_grad = False

fc_lr = 0.01
optimizer = optim.SGD(list(model.classifier.children())[6].parameters(), lr=fc_lr, momentum=0.9, weight_decay=5e-4)
best_prec1 = 0
nums_epoch = 200


def main():
    global best_prec1
    for epoch in xrange(1, nums_epoch + 1):
        train_model(model, train_loader, criterion, optimizer, exp_lr_scheduler, epoch)
        prec1 = test_model(model, val_loader, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(model, epoch, is_best)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=40):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('learning rate is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

        if idx % print_frequency == 0:
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

        if idx % print_frequency == 0:
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


def save_checkpoint(model, epoch, is_best):
    model_out_path = './result/model_epoch_{}.pth'.format(epoch)
    torch.save(model, model_out_path)
    print('checkpoint saved to {}'.format(model_out_path))
    if is_best:
        shutil.copy(model_out_path, './result/model_best.pth')


if __name__ == '__main__':
    main()

