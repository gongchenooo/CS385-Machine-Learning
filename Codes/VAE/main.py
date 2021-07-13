import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import itertools
from tqdm import tqdm
import VAE2
import matplotlib
#import VAE
parser = argparse.ArgumentParser()
matplotlib.use('Agg')
"""Check if a CUDA GPU is available, and if yes use it. Else use the CPU for computations."""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("Using %s for computation" % device)
use_conv = True

batch_size = 32  # number of inputs in each batch
epochs = 20  # times to run the model on complete data
lr = 1e-3  # learning rate
train_loss = []
rec_loss = []
KLD_loss = []

if use_conv:
    image_size = 64  # dimension of the image
    hidden_size = 1024  # hidden dimension
    latent_size = 32  # latent vector dimension
    train_data = datasets.SVHN(
        root='../../Data',
        split='train',
        download=True,
        transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor()]),
    )
    test_data = datasets.SVHN(
        root='../../Data',
        split="test",
        download=True,
        transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor()]),
    )

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

def show_images(images):
    images = torchvision.utils.make_grid(images)
    show_image(images)


def show_image(img):
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


vae = VAE2.DCVAE(image_channels=3, image_dim = image_size,
                     hidden_size=hidden_size, latent_size=latent_size,).to(device)

optimizer = optim.Adam(vae.parameters(), lr=lr)


vae.train()
for epoch in tqdm(range(epochs)):
    for i, (images, _) in enumerate(trainloader):
        images = images.to(device)
        optimizer.zero_grad()
        reconstructed_image, mean, log_var = vae(images)
        if use_conv:
            CE = F.binary_cross_entropy(reconstructed_image, images, reduction="sum")
        else:
            CE = F.binary_cross_entropy(
                reconstructed_image, images.view(-1, input_size), reduction="sum"
            )
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = CE + KLD
        loss.backward()
        train_loss.append(loss.item())
        rec_loss.append(CE.item())
        KLD_loss.append(KLD.item())
        optimizer.step()

        if i % 100 == 0:
            print("Loss:{:.3f}\tBCE:{:.3f}\tKLD:{:.3f}".format(loss.item() / len(images), CE.item()/len(images), KLD.item()/len(images)))

np.save('loss.npy', train_loss)
np.save('rec_loss.npy', rec_loss)
np.save('KLD_loss.npy', KLD_loss)
plt.plot(train_loss)
plt.show()
plt.savefig('loss.png')
torch.save(vae.state_dict(), "DCVAE.pt")
"""
Set the model to the evaluation mode. This is important otherwise you will get inconsistent results. Then load data from the test set.
"""
#vae = torch.load('DCVAE.pt')

vae.eval()
vectors = []
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        reconstructed_image, mean, log_var = vae(images)
        reconstructed_image = reconstructed_image.view(-1, 3, image_size, image_size)
        temp = list(zip(labels.tolist(), mean.tolist()))
        for x in temp:
            vectors.append(x)
        if i % 100 == 0:
            show_images(reconstructed_image.cpu())
            img_name = str(i).zfill(3)
            #torchvision.utils.save_image(reconstructed_image.cpu(), img_name)
            plt.savefig('reconstruct/'+img_name)
            plt.show()

labels, z_vectors = list(zip(*vectors))
z_vectors = torch.tensor(z_vectors)
U, S, V = torch.svd(torch.t(z_vectors))
C = torch.mm(z_vectors, U[:, :2]).tolist()
C = [x + [labels[i]] for i, x in enumerate(C)]
df = pd.DataFrame(C, columns=['x', 'y', 'label']) 
df.head()
fig = sns.lmplot( x="x", y="y", data=df, fit_reg=False, hue='label')
fig.savefig('latent_distribution.png', dpi=400)

torch.save(vae.state_dict(), "DCVAE.pt")
#vae.load_state_dict(torch.load("DCVAE.pt"))
