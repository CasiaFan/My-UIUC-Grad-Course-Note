import torch
from torchvision import models, datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# load transform
train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.12, 0.12, 0.08, 0.08),
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform =  transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

batch_size = 16

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
train_size = len(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
test_size = len(testset)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)



device = "cuda:0"
# load model 
model = models.resnet50(pretrained=False, num_classes=10).to(device)

# loss function
criterion = torch.nn.CrossEntropyLoss().to(device)
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# train the model
epochs = 200

train_loss = []
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # load to gpu
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    epoch_loss = running_loss / (i+1)
    train_loss.append(epoch_loss)
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {epoch_loss:.3f}')
    running_loss = 0.0
print('Finished Training')

print("Train loss: ", train_loss)
# save trained model 
model_path = './cifar_net.pth'
torch.save(model.state_dict(), model_path)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 5)  

ax.plot(np.arange(len(train_loss)), train_loss, "-o", markersize=5, linewidth=2)
ax.set_title("Epoch vs Training loss")
# ax.set_ylim([0.0, 2.3])
# ax[i].yaxis.set_ticks(np.arange(0.85, 1.01, 0.05))
ax.set_xticks(np.arange(0, len(train_loss), 25))
plt.xlabel("Epoch")
plt.ylabel("Average Training loss")
plt.tight_layout()
plt.savefig("train_loss.png")
# plt.show()

# test
model.load_state_dict(torch.load(model_path))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

