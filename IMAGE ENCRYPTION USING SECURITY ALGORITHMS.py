import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

gray_path = "/aflal.pc/input/image-colorization/l/gray_scale.npy"
gray = np.load(gray_path)[:10000]
gray = gray / 255
gray = gray.reshape((-1, 1, 224, 224))

def display(img):
    plt.set_cmap('gray')
    plt.imshow(img[0], cmap='gray')
    plt.show()

sample = gray[100]
display(sample)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 8, stride=2, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 700
batch_size = 128
size = gray.shape[0]
print_every = 100

gray_tensor = torch.from_numpy(gray).float()

for epoch in range(num_epochs):
    total_loss = 0
    steps = size // batch_size

    for i in range(0, size, batch_size):
        data = gray_tensor[i:i + batch_size]
        generated_data = model(data)
        loss = criterion(generated_data, data)
        total_loss += loss.item() * data.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/size:.6f}")
