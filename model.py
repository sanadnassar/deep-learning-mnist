import torch
import torch.nn as nn
import torch.nn.functional as F

EPOCH_BREAK_ACCURACY = 0.995  # Stop training if it reaches this accuracy
TEST_BATCH_SIZE = 1000  # Number of pictures we are using


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Conv2d(in, out, kernel, stride)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)  # Dropout(percentOfNeurons)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)  # Linear(in, out)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train_model(model, device, data_loader, loss_func, optimizer, num_epochs):
    train_loss, train_acc = [], []
    
    for epoch in range(num_epochs):
        runningLoss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            runningLoss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        # Compute epoch loss and accuracy
        epoch_loss = runningLoss / len(data_loader)
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        if epoch_acc >= EPOCH_BREAK_ACCURACY:
            print(f"Model has reached 99.5% accuracy, stopping training")
            return train_loss, train_acc  # Exit function instead of breaking only the inner loop

    return train_loss, train_acc


def test_model(model, data_loader, device=None):
    if device is None:
        device = torch.device("cpu")

    model.eval()
    test_loss = 0
    correct = 0

    loss_func = nn.CrossEntropyLoss()  # Ensure loss function is defined

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += loss_func(outputs, labels).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(data_loader)
    accuracy = correct / len(data_loader.dataset)
    
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    return test_loss, accuracy