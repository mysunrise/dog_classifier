from __future__ import print_function
import torch
import torchvision
import torchvision.models as models

vgg19 = models.vgg19(pretrained=True)
if torch.cuda.is_available():
    vgg19 = vgg19.cuda()
# print(vgg19)
for name, module in vgg19.named_children():
    print(name)
    if name in ['classifier']:
        for newname in vgg19.classifier.named_children():
            print(newname)
# vgg19.classifier[6]=torch.nn.Linear(4096,100)
print(vgg19.classifier[6])
'''
for model_name in models.__dict__:
    print(model_name)

for idx, m in enumerate(vgg19.named_modules()):
    print(idx, '->', m)
for name, param in vgg19.named_parameters():
    print(name, param.size())
for idx, key in enumerate(vgg19.state_dict()):
    print(key)
'''
for i in range(10):
    print(i)
