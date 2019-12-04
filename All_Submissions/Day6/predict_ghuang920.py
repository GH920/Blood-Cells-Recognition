import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2
import os
from torch.autograd import Variable

# os.system('sudo pip install cv2')

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform((x))

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# This predict is a dummy function, yours can be however you like as long as it returns the predictions in the right format
def predict(imglist):
    # On the exam, x will be a list of all the paths to the images of our held-out set
    images = []
    resizehere = (400, 300)
    for img_path in imglist:
        # Here you would write code to read img_path and preprocess it
        # img = cv2.cvtColor(cv2.resize(cv2.imread(img_path), resizehere), cv2.COLOR_RGB2GRAY)
        # _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # blurred = cv2.GaussianBlur(img, (3, 3), 0)
        # img = thresh * img
        img = cv2.resize(cv2.imread(img_path), resizehere)
        img = img.reshape(3, resizehere[1], resizehere[0]) / 255
        images.append(img)  # I am using 1 as a dummy placeholder instead of the preprocessed image
    x = torch.FloatTensor(np.array(images))
    x_trans = torch.zeros([x.shape[0], 3, 224, 224])
    for i, j in enumerate(x):
        x_trans[i] = data_transforms['val'](j)
    # Here you would load your model (.pt) and use it on x to get y_pred, and then return y_pred
    model = torchvision.models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier._modules['6'] = nn.Linear(4096, 7)
    model.load_state_dict(torch.load('model_ghuang920.pt'))
    # model.cuda()
    model.eval()

    # x = Variable(x).cuda()
    x = Variable(x_trans)
    y_pred = model(x).data
    return (y_pred>0).float()