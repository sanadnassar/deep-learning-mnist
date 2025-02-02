import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from torchvision import datasets, transforms
from model import CNN, train_model, test_model
from lib.util import header, plot_distribution, double_plot
import matplotlib as plt


SESSION_1_EPOCH_COUNT = 3
SESSION_2_EPOCH_COUNT = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000

def det_device_config(train_kwargs, test_kwargs):
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    if use_cuda:
        print('using cuda device')
        device = torch.device("cuda")
    elif use_mps:
        print('using mps')
        device = torch.device("mps")
    else:
        print('using cpu')
        device = torch.device("cpu")

    if use_cuda:
        train_kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True})
        test_kwargs.update({'num_workers': 1, 'pin_memory': True})

    return device, train_kwargs, test_kwargs  # Ensure return always happens

if __name__ == "__main__":
    train_kwargs = {"batch_size": BATCH_SIZE}
    test_kwargs = {"batch_size": TEST_BATCH_SIZE}
    device, train_kwargs, test_kwargs = det_device_config(train_kwargs, test_kwargs)

    model = CNN().to(device)  # Fixed 'Model' typo

    header("Loading datasets...")
    normalization_transform = transforms.ToTensor()  # Ensure transform exists
    train_data = datasets.MNIST('../data', train=True, download=True, transform=normalization_transform)
    test_data = datasets.MNIST('../data', train=False, transform=normalization_transform)
    print("Done loading datasets")

    plot_distribution('Distribution of Labels in Training Set', train_data)
    plot_distribution('Distribution of Labels in Testing Set', test_data)

    header("Training the model...")
    optimizer = RMSprop(model.parameters(), lr=LEARNING_RATE)
    train_loss, train_acc = train_model(
        model,
        device,
        data_loader=DataLoader(train_data, **train_kwargs),
        loss_func=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        num_epochs=SESSION_1_EPOCH_COUNT
    )
    print("Done training")

    header('Saving checkpoint 1...')
    checkpoint_1_path = "../checkpoint1.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_1_path)
    print("Saved")

    header('Loading checkpoint 1...')
    checkpoint = torch.load(checkpoint_1_path)
    new_model = CNN().to(device)
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_optimizer = RMSprop(new_model.parameters(), lr=LEARNING_RATE)
    new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded")

    print("Checking Accuracy...")
    old_test_loss, old_test_accuracy = test_model(model, DataLoader(test_data, **test_kwargs), device)
    new_test_loss, new_test_accuracy = test_model(new_model, DataLoader(test_data, **test_kwargs), device)
    print(f"Loaded Accuracy: {new_test_accuracy:.4f}, Expected Accuracy: {old_test_accuracy:.4f}")
    print("Done")

    header("Training the model...")
    train_loss_2, train_acc_2 = train_model(
        new_model,
        device,
        data_loader=DataLoader(train_data, **train_kwargs),
        loss_func=torch.nn.CrossEntropyLoss(),
        optimizer=new_optimizer,
        num_epochs=SESSION_2_EPOCH_COUNT
    )
    print('Done training')

    if train_loss_2 and train_acc_2:
        train_loss.extend(train_loss_2)
        train_acc.extend(train_acc_2)

    header("Saving the model to file...")
    MODEL_PATH = "../mnist_cnn.pt"
    torch.save(new_model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    double_plot(label1="Training Loss", data1=train_loss, label2="Training Accuracy", data2=train_acc)

    
    

    