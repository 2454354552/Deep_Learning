import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

train_ds = torchvision.datasets.MNIST('data',
                                      train=True,
                                      transform=transform,
                                      download=True)
dataloader = torch.utils.data.DataLoader(train_ds,
                                         batch_size=64,
                                         shuffle=True)

#定义生成器
#输入(100,)  输出(1,28,28)
#linear 100 -> 256
#linear 200 -> 512
#linear 512 -> 28*28
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )
    def forward(self, x):
        img = self.main(x)
        img = img.view(-1,28,28)
        return img

#定义判别器
#输入(1,28,28)  输出为二分类的概率值，输出使用sigmoid
#BCELoss计算交叉熵损失
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.main(x)
        return x

#初始化模型、优化器、损失函数
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gen = Generator().to(device)
dis = Discriminator().to(device)

d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)

loss_fn = nn.BCELoss()


#绘图函数
def gen_img_plot(model,  test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4,4))
    for i in range(prediction.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axis('off')
    plt.show()

test_input = torch.randn(16,100, device=device)

#GAN训练
D_loss = []
G_loss = []

# 训练循环
epochs = 100

for epoch in range(epochs):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    for step, (img, _) in enumerate(dataloader):
        #         print(img.shape[0])
        img = img.to(device)

        size = img.shape[0]
        random_noise = torch.randn(size, 100, device=device)

        # 判别器损失
        d_optim.zero_grad()
        real_output = dis(img)  # 判别器输入真实图片
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))  # 判别器在真实图像上的损失

        d_real_loss.backward()

        gen_img = gen(random_noise)
        fake_output = dis(gen_img.detach())  # 判别器输入生成图片
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        # 生成器损失
        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_fn(fake_output, torch.ones_like(fake_output))  # 生成器损失
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch:', epoch)
        if epoch == epochs-1:
            gen_img_plot(gen, test_input)