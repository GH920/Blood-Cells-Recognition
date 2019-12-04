# --------------------------------------------------------------------------------------------
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

# --------------------------------------------------------------------------------------------
# Hyper Parameters
num_epochs = 20
batch_size = 20
learning_rate = 0.001

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

# load data and create mini-batch data loader
torch.manual_seed(1122)
X, y = np.load('imgs.npy', allow_pickle=True), np.load('txts.npy', allow_pickle=True)
X = X.reshape(929, 3, 300, 400)/255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rgb_mean = np.mean(X_train, axis=(0,2,3))
rgb_std = np.std(X_train, axis=(0,2,3))
X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)
train_transforms = transforms.Compose([transforms.ToPILImage(),
                                       transforms.RandomRotation(30),
                                       # transforms.RandomResizedCrop(300),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       # transforms.Normalize(rgb_mean, rgb_std)]
                                      ])
# test_transforms = transforms.Compose([transforms.ToPILImage(),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize(rgb_mean, rgb_std)])

trainset = CustomTensorDataset(tensors=(X_train, y_train), transform=train_transforms)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# --------------------------------------------------------------------------------------------
testset = TensorDataset(X_test, y_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
# -----------------------------------------------------------------------------------
# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Dropout2d(0.2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2))

        self.fc1 = nn.Sequential(
            nn.Linear(37 * 50 * 32, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )

        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
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
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.7f'
                  % (epoch + 1, num_epochs, i + 1, len(y_train) // batch_size, loss.item()))
# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

criterion = nn.BCELoss()
val_loss = torch.Tensor([0])
count = 0
for images, labels in testloader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    predicted = (outputs.data.cpu()>0).float()
    val_loss += criterion(predicted, labels)
    count += 1
val_loss /= count
print('BCE Loss in test data', val_loss)
# -----------------------------------------------------------------------------------
# print('Test Accuracy of the model on the %d test images: %d %%' % (len(y_test), 100 * correct / total))
# -----------------------------------------------------------------------------------
# Save the Trained Model
torch.save(cnn.state_dict(), 'model_ghuang920_2.pt')