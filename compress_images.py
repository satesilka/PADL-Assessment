import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
            nn.Conv2d(192, 384, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU())

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(384 * 6 * 5, latent_dim)

        # Decoder
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 384 * 6 * 5),
            nn.ReLU(0.1))

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (384, 6, 5)),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc_enc(x)
        x = self.fc_dec(z)
        x = self.decoder(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode(images):
    model = Autoencoder(latent_dim=32).to(device)
    model.load_state_dict(torch.load("q6_model.pth", map_location=device))
    model.eval()

    images = images.to(device)
    with torch.no_grad():
        x = model.encoder(images)
        x = model.flatten(x)
        z = model.fc_enc(x)
    return z.cpu()

def decode(latents):
    model = Autoencoder(latent_dim=32).to(device)
    model.load_state_dict(torch.load("q6_model.pth", map_location=device))
    model.eval()

    latents = latents.to(device)
    with torch.no_grad():
        x = model.fc_dec(latents)
        x = model.decoder(x)
    return x.cpu()



