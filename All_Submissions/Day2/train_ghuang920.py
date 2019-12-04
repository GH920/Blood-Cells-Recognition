# --------------------------------------------------------------------------------------------
import torch
import numpy as np
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------------------------
# Hyper Parameters
num_epochs = 20
batch_size = 20
learning_rate = 0.001

# load data and create mini-batch data loader
torch.manual_seed(1122)
X, y = np.load('imgs.npy', allow_pickle=True), np.load('txts.npy', allow_pickle=True)
X = X.reshape(929, 3, 300, 400)/255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
trainset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# --------------------------------------------------------------------------------------------
testset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
# -----------------------------------------------------------------------------------
# CNN Model (2 conv layer)
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
        # out = torch.sigmoid(out)
        return out
# -----------------------------------------------------------------------------------
cnn = CNN()
cnn.cuda()
# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        labels = Variable(labels).cuda()
        images = Variable(images).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % (100//batch_size) == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(y_train) // batch_size, loss.item()))
# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in testloader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    predicted = (outputs.data.cpu()>0).float()
    total += labels.size(0)
    correct += (predicted == labels).sum()/7
# -----------------------------------------------------------------------------------
print('Test Accuracy of the model on the %d test images: %d %%' % (len(y_test), 100 * correct / total))
# -----------------------------------------------------------------------------------
# Save the Trained Model
torch.save(cnn.state_dict(), 'model_ghuang920.pt')