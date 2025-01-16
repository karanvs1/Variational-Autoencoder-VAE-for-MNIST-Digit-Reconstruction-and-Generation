import torch
import torch.nn as nn
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VAE model class
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):    
        z = self.encoder(x)

        # split the latent vector into mean and variance
        mu = z[:, :self.latent_size]
        log_var = z[:, self.latent_size:]
        std = torch.exp(0.5 * log_var)

        # sampling and reparameterization trick
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return self.decoder(z), mu, log_var
    
    def generate(self, z):
        return self.decoder(z)
    
if __name__ == '__main__':
    model = VAE(input_size=784, hidden_size=128, latent_size=64).to(device)

    # load model
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    # test
    x = torch.randn(64, 784).to(device)
    y, mu, log_var = model(x)
    print(y.shape, mu.shape, log_var.shape)

    # generate
    z = torch.randn_like(log_var)
    y = model.generate(z)
    
    # save image
    y = y.view(-1, 1, 28, 28)
    save_image(y, "image.png")