import torch
import torch.nn as nn
import time
import tracemalloc
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import RealTime_visualization



def train_and_evaluate(model, optimizer, scheduler, train_loader, test_loader, device, epochs=5, show_plot=False):
    criterion = nn.CrossEntropyLoss()
    model.train()

    losses = []
    tracemalloc.start()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        losses.append(epoch_loss / len(train_loader))

        if show_plot:
            # Real-time plotting
            RealTime_visualization(losses, epoch)

    end_time = time.time() - start_time
    memory_used = tracemalloc.get_traced_memory()[1] / (1024 ** 2)  # Convert to MB
    tracemalloc.stop()

    accuracy = evaluate(model=model, test_loader=test_loader, device=device)

    return losses, accuracy, end_time, memory_used



def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
