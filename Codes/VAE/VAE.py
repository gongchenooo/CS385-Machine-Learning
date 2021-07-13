import torch.autograd
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

z_dimension = 2

def dataloader(batch_size):
    data_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.SVHN(root='../../Data', split='train', transform=data_tf, download=True)
    test_dataset = datasets.SVHN(root='../../Data', split='test', transform=data_tf, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    in_channels = 3
    num_classes = 10
    return train_loader, test_loader, in_channels, num_classes

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 32->16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16->8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8->4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.encoder_fc1 = nn.Linear(128 * 4 * 4, z_dimension)
        self.encoder_fc2 = nn.Linear(128 * 4 * 4, z_dimension)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(z_dimension, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, out_channels=3, kernel_size=3, padding=1),
            nn.Tanh())


    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 128, 4, 4)
        out3 = self.decoder(out3)
        out3 = self.final_layer(out3)
        return out3, mean, logstd

def loss_function(recon_x, x, mean, std, KLD_weight=1):
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var = torch.pow(torch.exp(std), 2)
    KLD = torch.mean(-0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var, dim=1), dim=0)
    return MSE + KLD_weight*KLD, MSE, KLD

# 创建对象
vae = VAE().to(device)
num_epoch = 15
# vae.load_state_dict(torch.load('./VAE_z2.pth'))
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0)\

train_loader, test_loader, _, _ = dataloader(64)
###########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (img, _) in enumerate(train_loader):
        num_img = img.size(0)
        # view()函数作用把img变成[batch_size,channel_size,784]
        img = img.view(num_img, 3, 32, 32).to(device)  # 将图片展开为28*28=784
        x, mean, std = vae(img)  # 将真实图片放入判别器中
        loss, MSE, KDL = loss_function(x, img, mean, std)
        vae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        loss.backward()  # 将误差反向传播
        vae_optimizer.step()  # 更新参数
        # try:
        if (i + 1) % 100 == 0:
            print('Epoch[{}/{}],vae_loss:{:.6f} '.format(
                epoch, num_epoch, loss.item(),
            ))

        if i == 0:
            real_images = make_grid(img.cpu(), nrow=8, normalize=True).detach()
            save_image(real_images, './img_VAE/real_images.png')
            sample = torch.randn(64, 2).to(device)
            output = vae.decoder_fc(sample)
            output = vae.decoder(output.view(output.shape[0], 124, 4, 4))
            output = vae.decoder(output)
            output = vae.final_layer(output)
            fake_images = make_grid(x.cpu(), nrow=8, normalize=True).detach()
            save_image(fake_images, './img_VAE/fake_images-{}.png'.format(epoch))
# 保存模型
torch.save(vae.state_dict(), './VAE.pth')