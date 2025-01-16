import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import VAE

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Config
config = {
    "batch_size": 128,
    "num_epochs": 25,
    "lr": 1e-3,
    "input_size": 784,
    "hidden_size": 128,
    "latent_size": 64,
    "alpha": 0.001
}

# Dataset and DataLoader
mnist_train = dset.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dset.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(mnist_train, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=config["batch_size"], shuffle=False)

# Model
model = VAE(config["input_size"], config["hidden_size"], config["latent_size"]).to(device)

# Optimizer and Loss
optim = optim.Adam(model.parameters(), lr=config["lr"])
criterion = nn.MSELoss()

# Loss
def compute_loss(y, data, mu, log_var):
    reconstruction_loss = criterion(y, data)
    kl_divergence = 0.5 * (torch.sum(-log_var + torch.exp(log_var) + mu**2 - 1, axis=1)).mean()
    return reconstruction_loss + config["alpha"] * kl_divergence

# Train
for epoch in range(config["num_epochs"]):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, config["input_size"]).to(device)
        optim.zero_grad()
        y, mu, log_var = model(data)
        loss = compute_loss(y, data, mu, log_var)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()

    print(f"Epoch: {epoch + 1} | Train Loss: {epoch_loss / len(train_loader)}")
    
    # Test
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.view(-1, config["input_size"]).to(device)
            y, mu, log_var = model(data)
            loss = compute_loss(y, data, mu, log_var)
            test_loss += loss.item()

    print(f"Epoch: {epoch + 1} | Test Loss: {test_loss / len(test_loader)}")

# Save model    
torch.save(model.state_dict(), "model.pt")

# Generate images
with torch.no_grad():
    model.eval()
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.view(-1, config["input_size"]).to(device)
        y, mu, log_var = model(data)
        y = y.view(-1, 1, 28, 28)
        data = data.view(-1, 1, 28, 28)
        # flip the color of the ground truth for visualization
        data = 1 - data
        # stack them side by side
        img = torch.cat([data, y], dim=3)
        if not os.path.exists("image"):
            os.makedirs("image")
        save_image(img, f"image/{batch_idx}.png")
        break



