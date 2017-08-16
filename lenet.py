from network import *

import torchvision
from torchvision import transforms

transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root = './data', train = True, 
	download = False, transform = transform)
testset = torchvision.datasets.CIFAR10(root = './data', train = False, 
	download=False, transform = transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = 16, 
	shuffle = True, num_workers = 2)
test_loader = torch.utils.data.DataLoader(testset, batch_size = 16, 
	shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
		   'dog', 'frog', 'horse', 'ship', 'truck')

nums_epoch = 10
print_prequency = 500

features = [	nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
				nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2)]
classifier = [	nn.Linear(16*5*5, 120), nn.ReLU(),
				nn.Linear(120, 84), nn.ReLU(),
				nn.Linear(84, 10)]
criterion = nn.CrossEntropyLoss()
lenet = network(features, classifier, criterion, False, True, 0.01)

lenet.train_dataset(train_loader, nums_epoch, print_prequency)
lenet.test_dataset(test_loader)

'''
for epoch in xrange(10):
	lenet.state.reset()
	for i, (x, y) in enumerate(trainloader, 0):
		lenet.train_model(Variable(x), Variable(y))
		if i % 500 == 0:
			print('\nepoch: %d batch: %d / %d' %(epoch+1, i, len(trainloader)))
			lenet.state.show()

print('\nTraining complete')

lenet.state.reset()
for (x, y) in testloader:
	lenet.test_model(Variable(x), Variable(y))
lenet.state.show()

print('\nTesting complete')
'''



