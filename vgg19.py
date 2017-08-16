from __future__ import print_function
from __future__ import division

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

import network
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-c', type = int, default = 10, help = 'number of class')
parser.add_argument('-e', type = int, default = 5, help = 'number of epoch')
parser.add_argument('-p', type = int, default = 100, help = 'print frequency')
parser.add_argument('-b', type = int, default = 32, help = 'batch size')
parser.add_argument('-l', type = float, default = 0.01, help = 'learning rate')
parser.add_argument('-ldr', type = float, default = 0.1, help = 'learning rate descending rate')
parser.add_argument('-lde', type = int, default = 5, help = 'learning rate decay epoch')

args = parser.parse_args()
num_classes = args.c
nums_epoch = args.e
print_frequency = args.p
batch_size = args.b
lr = args.l
lr_descending_rate = args.ldr
lr_decay_epoch = args.lde

print ('Number of class: %s' % args.c)
print ('Number of epoch: %s' % args.e)
print ('Learning rate: %f' % args.l)
print ('Learning rate descending rate: %f' % args.ldr)
print ('Learning rate decay epoch: %s' % args.lde)

train_dir = '../train'
test_dir = '../train'

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								  std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder(train_dir, transforms.Compose([
		transforms.RandomSizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalizer
	])),
	batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder(test_dir, transforms.Compose([
		transforms.Scale(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalizer
	])),
	batch_size=batch_size, shuffle=False)

pretrained_model = models.vgg19(pretrained = True)
features = list(pretrained_model.features.children())
classifier = list(pretrained_model.classifier.children())[:-1]
classifier.append(nn.Linear(4096, num_classes))
criterion = nn.CrossEntropyLoss()
vgg19 = network(features, classifier, criterion, 
	False, True, 
	lr, lr_descending_rate, lr_decay_epoch)
vgg19.train_dataset(train_loader, nums_epoch, print_frequency)
vgg19.test_dataset(test_loader)


'''
def save_checkpoint(model, epoch, is_best):
	model_out_path = './result/model_epoch_{}.pth'.format(epoch)
	torch.save(model, model_out_path)
	print('checkpoint saved to {}'.format(model_out_path))
	if is_best:
		shutil.copy(model_out_path, './result/model_best.pth')

'''

