import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# define related variables
batch_size = 32
num_classes = 10
learning_rate = 0.001
num_epochs = 10

# decide which device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the dataset and preprocess data.
train_dataset = torchvision.datasets.MNIST(root = './data',
                                           train = True,
                                           transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                           download = True)
test_dataset = torchvision.datasets.MNIST(root = './data',
                                           train = False,
                                           transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                           download = True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)
# defining the convolutional nerual network
class LeNet5(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        # can both use separate declaration of activation function and layer, or
        # can combine them using nn.Sequential tom better organise.
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
    def forward(self, x):
        # this is forward function, defines how the nuerual network produce output 
        # by input given by the batch of dataset in each epoch.
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # reshape the output into one-dimensional vector, for FC to use
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# prepare for training

model = LeNet5(num_classes=num_classes).to(device)
# setting the loss function (to evaluate the efficiency of CNN)
cost = nn.CrossEntropyLoss()
# setting the optimiser with the model parameters and learning rate.
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
#this is defined to print how many steps are remaining when training
total_step = len(train_loader)

# start training

for epoch in range(num_epochs):
    for i , (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = cost(outputs, labels)

        # backward and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i + 1) % 400 == 1:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# start testing
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # find the max element in dimension given arguement (this case it's 1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
	 

            


        




