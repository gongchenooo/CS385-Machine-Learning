import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from DCGan import weights_init, Generator, Discriminator

seed = 123
random.seed(seed)
torch.manual_seed(seed)

batch_size=128
epochs = 20
save_per_epoch = 2
lr = 2e-4
z_dim = 100
g_feature_num = 64
d_feature_num = 64


device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
dataloader = datasets.SVHN(
        root='../../Data',
        split='train',
        download=True,
        transform=transforms.Compose([transforms.Resize(64),
                                      transforms.CenterCrop(64),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ]),
    )
dataloader = DataLoader(dataloader, batch_size=128, shuffle=True)
g_model = Generator(z_dim=z_dim, g_feature_num=g_feature_num).to(device)
d_model = Discriminator(d_feature_num=d_feature_num).to(device)
g_model.apply(weights_init)
d_model.apply(weights_init)

print('Generator model:', g_model)
print('Discriminator model:', d_model)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)
d_optimizer = optim.Adam(d_model.parameters(), lr=lr)
g_optimizer = optim.Adam(g_model.parameters(), lr=lr)


img_list = []
g_loss_list = []
d_loss_list = []

iters = 0

for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        real_data = data[0].to(device)
        batch_size = real_data.size(0)

        d_model.zero_grad()
        label = torch.full((batch_size,), 1, device=device, dtype=float)
        output = d_model(real_data).view(-1)
        errD_real = criterion(output.float(), label.float())
        errD_real.backward()

        D_x = output.mean().item()
        
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake_data = g_model(noise)
        label.fill_(0)
        output = d_model(fake_data.detach()).view(-1)
        errD_fake = criterion(output.float(), label.float())
        errD_fake.backward()
        D_G_1 = output.mean().item()
        errD = errD_real + errD_fake
        d_optimizer.step()

        
        g_model.zero_grad()
        label.fill_(1)
        output = d_model(fake_data).view(-1)
        errG = criterion(output.float(), label.float())
        errG.backward()
        D_G_2 = output.mean().item()
        g_optimizer.step()

        g_loss_list.append(errG.item())
        d_loss_list.append(errD.item())

        if i % 50 == 0:
            print(torch.cuda.is_available())
            print('[{}/{}][{}/{}]\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tD(x): {:.4f}\tD(G(z)): {:.4f} -> {:.4f}'.format(
                epoch, epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_1, D_G_2))

        if (iters % 100 == 0):
            with torch.no_grad():
                fake_data = g_model(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))
        iters += 1



np.save("G_loss.npy", g_loss_list)
np.save("D_loss.npy", d_loss_list)
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_loss_list, label="G")
plt.plot(d_loss_list, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss.png')
plt.show()


fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save('SVHN.gif', dpi=80, writer='imagemagick')