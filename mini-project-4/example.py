#%%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    raise RuntimeWarning("You are using a CPU, it will take a very long time to train")

print(device)

normalize = transforms.Normalize(mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375])


transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
)
trainset = torchvision.datasets.ImageFolder(root=TRAIN_SET, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
testset = torchvision.datasets.ImageFolder(root=TEST_SET, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
vgg16 = models.vgg16(pretrained=True)
vgg16.to(device)

print(vgg16)

# change the number of classes
vgg16.classifier[6].out_features = 10
# freeze convolution weights
for param in vgg16.features.parameters():
    param.requires_grad = False

# optimizer
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.01, momentum=0.9)
# loss function
criterion = nn.MSELoss()



# validation function
def validate(model, test_dataloader):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    
    for int, data in enumerate(test_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        output = model(data)
        loss = criterion(output, target)

        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()

    val_loss = val_running_loss / len(test_dataloader.dataset)
    val_accuracy = 100.0 * val_running_correct / len(test_dataloader.dataset)

    return val_loss, val_accuracy


# training function
def fit(model, train_dataloader):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / len(train_dataloader.dataset)
    train_accuracy = 100.0 * train_running_correct / len(train_dataloader.dataset)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")

    return train_loss, train_accuracy


train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
start = time.time()
for epoch in range(10):
    train_epoch_loss, train_epoch_accuracy = fit(vgg16, trainloader)
    val_epoch_loss, val_epoch_accuracy = validate(vgg16, testloader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
end = time.time()
print((end - start) / 60, "minutes")


#%%
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color="green", label="train accuracy")
plt.plot(val_accuracy, color="blue", label="validataion accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.show()

#%%
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color="orange", label="train loss")
plt.plot(val_loss, color="red", label="validataion loss")
plt.legend()
plt.savefig("loss.png")
plt.show()