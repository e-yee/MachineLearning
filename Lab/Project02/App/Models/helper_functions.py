import matplotlib.pyplot as plt
import numpy as np
import torch

def draw_epochs_losses(epochs_losses):
    epochs = epochs_losses[:, 0]
    losses = epochs_losses[:, 1]

    plt.figure(figsize=(8,6))
    plt.plot(epochs, losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def scale_data(data):
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    return (data - mean) / std


def fit(model, criterion, optimizer, train_data, train_target, epochs=1000, n_iter_not_change=5, lr_decay=0.1):
    old_loss = 0
    n_iter = 0
    epochs_losses = []
    
    for epoch in range(epochs):
        y_pred = model(train_data)
        loss = criterion(y_pred, train_target)
        old_loss = loss if (old_loss != loss) else old_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, LR: {lr:.5f}")
            epochs_losses.append([epoch + 1, loss.item()])
            
        if (old_loss == loss):
            n_iter += 1
            if (n_iter >= n_iter_not_change):
                n_iter = 0
                optimizer.param_groups[0]["lr"] *= lr_decay
    
    return model, np.array(epochs_losses)


def evaluate_model(model, criterion, test_data, test_target):
    with torch.no_grad():
        y_pred = model(test_data)
        
    loss = criterion(y_pred, test_target)
    return loss