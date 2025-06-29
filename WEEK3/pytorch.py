import torch                              
import torch.nn as nn                    
import torch.optim as optim             
import torch.nn.functional as F          
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader   

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        # Define the second fully connected layer:
        # Input features from the previous layer and output should be 10 classes (digits 0-9).
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Flatten the input x from shape [batch_size, 1, 28, 28] to [batch_size, 28*28]
        x = x.view(-1, 28 * 28)
        # Apply the first layer then a ReLU activation function to introduce non-linearity.
        x = F.relu(self.fc1(x))
        # Pass the output through the second layer which outputs the logits for 10 classes.
        x = self.fc2(x)
        return x

# Instantiate the model.
model = SimpleNN()

# Define the optimizer and the loss function.
# - Adam is a popular optimizer that adapts the learning rate.
# - CrossEntropyLoss combines softmax and negative log likelihood loss, appropriate for multi-class classification.
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop: iterate over the dataset multiple times (epochs).
for epoch in range(5):  # Train for 5 epochs.
    model.train()  # Set the model to training mode.
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()          # Clear gradients from the previous iteration.
        output = model(data)           # Forward pass: compute the model output for the current batch.
        loss = criterion(output, target)  # Compute the loss between prediction and true labels.
        loss.backward()                # Backward pass: compute the gradients.
        optimizer.step()               # Update model parameters using the gradients.

        # Print current training progress every 100 batches.
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()}")

    # Evaluation loop to assess model performance on the test dataset.
    model.eval()   # Set the model to evaluation (inference) mode.
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient calculation for inference.
        for data, target in test_loader:
            output = model(data)                # Compute output for test data.
            test_loss += criterion(output, target).item()  # Sum up batch losses.
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max logit (predicted class).
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions.

    test_loss /= len(test_loader.dataset)  # Compute average loss.
    accuracy = 100. * correct / len(test_loader.dataset)  # Calculate accuracy in percentage.
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
