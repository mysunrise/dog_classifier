# dog_classifier
This project aims to solve the problem described in http://js.baidu.com/. The contest aims to classify the pet dog.

For a baseline, we use VGG19 or ResNet152 as the pretrained model.
We can fine-tune the model by retraining the last fully connected layer and the convolutional layers.

The data/make_imagefolder.py aims to make the dataset satisfied with the class ImageFolder in pytorch.
We can use main.py to train the model.

This is my first pytorch projects, the architecture of the code is maily referenced from https://github.com/pytorch/examples/tree/master/imagenet.
