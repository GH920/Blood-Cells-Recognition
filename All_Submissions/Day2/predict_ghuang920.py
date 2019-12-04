import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
from torch.autograd import Variable

# os.system('sudo pip install cv2')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(75 * 100 * 32, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# This predict is a dummy function, yours can be however you like as long as it returns the predictions in the right format
def predict(imglist):
    # On the exam, x will be a list of all the paths to the images of our held-out set
    images = []
    resizehere = (400, 300)
    for img_path in imglist:
        # Here you would write code to read img_path and preprocess it
        img = cv2.resize(cv2.imread(img_path), resizehere)
        img = img.reshape(3, resizehere[1], resizehere[0]) / 255
        images.append(img)  # I am using 1 as a dummy placeholder instead of the preprocessed image
    x = torch.FloatTensor(np.array(images))
    # Here you would load your model (.pt) and use it on x to get y_pred, and then return y_pred
    model = CNN()
    model.load_state_dict(torch.load('model_ghuang920.pt'))
    # model.cuda()
    model.eval()

    # x = Variable(x).cuda()
    x = Variable(x)
    y_pred = model(x).data
    return (y_pred>0).float()