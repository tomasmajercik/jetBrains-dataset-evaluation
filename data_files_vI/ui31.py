import torch
import torch.nn as nn  # all neural network modules
import torch.optim as optim  # optimization algo
from torch.utils.data import DataLoader  # easier dataset management, helps create mini batches
import torchvision.datasets as datasets  # standard datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.metrics import confusion_matrix


from NeuralNwModule import NN  # Your custom neural network module

# Initialize variables for best accuracy tracking
best_accuracy = 0
best_epoch = 0

# Hyperparameters and constants
batch_size = 64
input_size = 784  # 28*28
num_outputs = 10  # numbers from 0-9
learning_rate = 0.001
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Set device
device = torch.device('cpu')

# Initialize neural network
network = NN(input_size, num_outputs).to(device)

# Loss function
misstake = nn.CrossEntropyLoss()

# Choose optimizer
print("Choose optimizer:")
print("1: SGD (Stochastic Gradient Descent)")
print("2: SGD with Momentum")
print("3: Adam (Adaptive Moment Estimation)")
choice = input("Enter your choice (1/2/3): ").strip().lower()

if choice == "1" or choice == "sgd":
    optimizer = optim.SGD(network.parameters(), lr=learning_rate)
elif choice == "2" or choice == "sgdm":
    momentum_to_use = float(input("Enter the momentum to use (0.9 recommended): "))
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum_to_use)
elif choice == "3" or choice == "adam":
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
else:
    raise ValueError("Invalid optimizer choice! Please restart and choose a valid option.")

# Track losses and accuracy
train_losses = []
test_losses = []
test_accuracies = []

# confusion matrix generation from chatGPT
def plot_confusion_matrix(y_true, prediction, classes, epoch=None):
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true, prediction)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix" + (f" (Epoch {epoch + 1})" if epoch is not None else ""))
    plt.colorbar()

    # Add labels to the axes
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add numbers inside the cells
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], "d"), ha="center", va="center", color="black")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

# Function to calculate accuracy and loss
def check_accuracy_and_loss(loader, nw, ep=0):
    global best_accuracy
    global best_epoch

    num_correct = 0
    num_samples = 0
    total_loss = 0
    nw.eval()

    all_preds = []  # Collect all predictions
    all_labels = []  # Collect all ground-truth labels

    with torch.no_grad():  # Don't compute gradients
        for x, y in loader:
            x = x.to(device) # sent to CPU for computation
            y = y.to(device)
            x = x.reshape(x.shape[0], -1) # converts 2d tensor to 1d tensor

            scores = nw(x) # forward pass
            loss = misstake(scores, y)  # Compute loss using CrossEntropyLoss
            total_loss += loss.item() # Accumulate loss for the batch

            # evaluate, check predictions and count them
            _, predicted = scores.max(1)  # Predicted labels
            num_correct += (predicted == y).sum()
            num_samples += predicted.size(0)

            if not loader.dataset.train:  # For confusion matrix, collect only test data
                all_preds.append(predicted.cpu())
                all_labels.append(y.cpu())

    accuracy = 100 * num_correct / num_samples
    average_loss = total_loss / len(loader)

    if loader.dataset.train:
        train_losses.append(average_loss)  # Append training loss
        print(f"Training Accuracy: {accuracy:.2f}%, Loss: {average_loss:.4f}")
    else:
        test_losses.append(average_loss)  # Append test loss
        test_accuracies.append(accuracy)  # Append test accuracy

        # Update best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = ep

        print(f"Test Accuracy: {accuracy:.2f}%, Loss: {average_loss:.4f}")

        # Plot confusion matrix
        all_preds = torch.cat(all_preds).numpy() # list of predictions
        all_labels = torch.cat(all_labels).numpy() # contains 1D tensors of targets
        plot_confusion_matrix(all_labels, all_preds, classes=[str(i) for i in range(10)], epoch=ep)

    nw.train()


# Train network
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        targets = target.to(device)

        # Flatten data
        data = data.reshape(data.shape[0], -1)

        # Forward pass
        scores = network(data) # Output of the network
        loss = misstake(scores, targets)

        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward() # compute gradients

        # Gradient descent step - update weights
        optimizer.step()

    # Check accuracy and loss after each epoch
    check_accuracy_and_loss(train_loader, network, epoch)
    check_accuracy_and_loss(test_loader, network, epoch)

# Plotting the Loss and Accuracy graphs
plt.figure(figsize=(12, 6))

# Subplot 1: Cross-Entropy Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", color="blue")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("Loss Over Epochs")
plt.legend()

# Subplot 2: Test Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy", color="green")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

print(f"\nBest accuracy on test data: {best_accuracy:.2f}% in epoch {best_epoch+1}")
