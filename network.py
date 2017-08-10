from __future__ import print_function
#from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import torchvision
from torchvision import transforms

import time

class AverageMeter(object):
	def __init__(self, func1 = None, func2 = None):
		self.avg = 0
		self.sum = 0
		self.val = 0
		self.count = 0
		self.func1 = func1
		self.func2 = func2
		
	def reset(self):
		self.avg = 0
		self.sum = 0
		self.val = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class NetState(object):
	def __init__(self):
		self.state = {}

	def reset(self):
		for i in self.state.values():
			i.reset()

	def show(self):
		for i in self.state:
			print(i + ': val: ' + '%.3f'%self.state[i].val + ' avg: ' + '%.3f'%self.state[i].avg)

	def update_one(self, name, val, n=1):
		self.state[name].update(val, n)

	def add(self, name, func1, func2):
		self.state[name] = AverageMeter(func1, func2)

	def update(self, net, x, y):
		for name in self.state:
			self.update_one(name, self.state[name].func1(net, x, y), self.state[name].func2(net, x, y))

class network(nn.Module):
	def __init__(self, features, classifier, criterion,
					   train_features = True, train_classifier = True,
					   init_lr = 0.01, lr_descending_rate = 0.1, lr_decay_epoch = 5):
		super(network, self).__init__()
		self.features = nn.Sequential(*features)
		self.classifier = nn.Sequential(*classifier)
		self.criterion = criterion
		self.optimizer = optim.SGD(self.classifier.parameters(), lr = init_lr, momentum = 0.9)
		self.init_lr = init_lr
		self.lr_descending_rate = lr_descending_rate
		self.lr_decay_epoch = lr_decay_epoch

		self.state = NetState()
		self.is_cuda = torch.cuda.is_available()
		if self.is_cuda:
			self.cuda()

		if not train_features:
			for i in self.features.children():
				for j in i.parameters():
					j.requires_grad = False

		if not train_classifier:
			for i in self.classifier.children():
				for j in i.parameters():
					j.requires_grad = False

		self.state.add('time', lambda net, x, y: time.time() - net.time, lambda net, x, y: 1)
		self.state.add('loss', lambda net, x, y: net.loss.data[0], lambda net, x, y: net.batch_size)
		self.state.add('top1', 
			lambda net, x, y: torch.sum(torch.max(net.y_, 1)[1] == y).data[0] * 1.0 / net.batch_size,
			lambda net, x, y: net.batch_size)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def set_lr(self, init_lr, lr_descending_rate, lr_decay_epoch):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = init_lr
		self.init_lr = init_lr
		self.lr_descending_rate = lr_descending_rate
		self.lr_decay_epoch = lr_decay_epoch

	def test_model(self, x, y, is_test = True):
		if is_test:
			self.eval()
		else:
			self.train()
		self.batch_size = y.size(0)
		self.time = time.time()

		self.y_ = self.forward(x)
		self.loss = self.criterion(self.y_, y)

		if is_test:
			self.state.update(self, x, y)

	def train_model(self, x, y):
		self.test_model(x, y, False)
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()
		self.state.update(self, x, y)

	def train_dataset(self, train_loader, nums_epoch, print_frequency):
		print('\nTraining start')
		for epoch in xrange(nums_epoch):
			self.state.reset()
			self.exp_lr_scheduler(epoch)
			for i, (x, y) in enumerate(train_loader):
				self.train_model(Variable(x).cuda(), Variable(y).cuda())
				if i % print_frequency == 0:
					print('\nepoch: %d batch: %d / %d' %(epoch, i, len(train_loader)))
					self.state.show()
		print('\nTraining complete')

	def test_dataset(self, test_loader):
		print('\nTesting start')
		self.state.reset()
		for (x, y) in test_loader:
			self.test_model(Variable(x).cuda(), Variable(y).cuda())
		self.state.show()
		print('\nTesting complete')

	def exp_lr_scheduler(self, epoch):
		if epoch != 0 and epoch % self.lr_decay_epoch == 0:
			self.init_lr *= self.lr_descending_rate
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = self.init_lr
			print('Learning rate is set to {}, epoch {}'.format(self.init_lr, epoch))

