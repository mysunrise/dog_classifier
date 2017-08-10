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

from network import *

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
	batch_size=32, shuffle=True)

test_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder(test_dir, transforms.Compose([
		transforms.Scale(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalizer
	])),
	batch_size=32, shuffle=False)

num_classes = 12
nums_epoch = 5
lr = 0.01
lr_decay_epoch = 40
print_frequency = 100

pretrained_model = models.vgg19(pretrained = True)
features = list(pretrained_model.features.children())
classifier = list(pretrained_model.classifier.children())[:-1]
classifier.append(nn.Linear(4096, num_classes))
criterion = nn.CrossEntropyLoss()
vgg19 = network(features, classifier, criterion, False, True, 0.01, 0.1, 2)
vgg19.train_dataset(train_loader, nums_epoch, print_frequency)
vgg19.test_dataset(test_loader)

'''
ignored_params = list(map(id, list(model.classifier.children())[6].parameters()))
base_params = filter(lambda x: id(x) not in ignored_params, model.parameters())
for param in base_params:
	param.requires_grad = False
optimizer = optim.SGD(list(model.classifier.children())[6].parameters(), lr=fc_lr, momentum=0.9, weight_decay=5e-4)


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

def save_checkpoint(model, epoch, is_best):
	model_out_path = './result/model_epoch_{}.pth'.format(epoch)
	torch.save(model, model_out_path)
	print('checkpoint saved to {}'.format(model_out_path))
	if is_best:
		shutil.copy(model_out_path, './result/model_best.pth')

'''

